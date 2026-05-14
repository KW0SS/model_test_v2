from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
REPORT_DIR = ROOT_DIR / "보고서"
DEFAULT_REPORT_PATH = REPORT_DIR / "raw_xgboost_학습결과_2026_0513.md"
DEFAULT_GROUP_COLUMN = "종목코드"
LABEL_COLUMN = "label"
RANDOM_STATE = 42
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)
METADATA_COLUMNS = {"기업상태", "기업명", "기업코드", "연도", "종목코드", "보고기간", "산업군"}
PERIOD_FEATURE_COLUMNS = {"period_Q1", "period_H1", "period_Q3", "period_ANNUAL"}

sys.path.append(str(MODEL_DIR))
from model_20260402_213421 import split_group_dataset, validate_group_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="raw 데이터 기반 Group Split XGBoost 실험")
    parser.add_argument("--data-path", type=Path, default=None, help="raw_모델학습용 CSV 경로")
    parser.add_argument("--group-column", type=str, default=DEFAULT_GROUP_COLUMN, help="group split 기준 컬럼")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="학습 데이터 비율")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="검증 데이터 비율")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="테스트 데이터 비율")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="학습 보고서 저장 경로")
    parser.add_argument("--timestamp", type=str, default=None, help="결과 파일명에 사용할 timestamp")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def find_latest_raw_training_csv() -> Path:
    candidates = sorted(
        (ROOT_DIR / "재무비율").glob("raw_모델학습용_*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("재무비율 폴더에서 raw_모델학습용_*.csv 파일을 찾지 못했습니다.")
    return candidates[0]


def load_training_dataframe(data_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series, pd.DataFrame]:
    dataframe = pd.read_csv(data_path, encoding="utf-8-sig")
    if LABEL_COLUMN not in dataframe.columns:
        raise KeyError(f"label 컬럼이 없습니다: {data_path}")
    if DEFAULT_GROUP_COLUMN not in dataframe.columns:
        raise KeyError(f"종목코드 컬럼이 없습니다: {data_path}")

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in METADATA_COLUMNS and column != LABEL_COLUMN
    ]
    x_data = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")
    y_data = pd.to_numeric(dataframe[LABEL_COLUMN], errors="coerce").astype(int)
    groups = dataframe[DEFAULT_GROUP_COLUMN].astype(str).str.strip().str.zfill(6)
    return x_data, y_data, feature_columns, groups, dataframe


def get_positive_proba(model: XGBClassifier, x_eval: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x_eval)[:, 1]


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
        }
        if (
            metrics["f1"] > best_metrics["f1"]
            or (metrics["f1"] == best_metrics["f1"] and metrics["recall"] > best_metrics["recall"])
            or (
                metrics["f1"] == best_metrics["f1"]
                and metrics["recall"] == best_metrics["recall"]
                and metrics["precision"] > best_metrics["precision"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if y_true.nunique() > 1 else None,
        "pr_auc": average_precision_score(y_true, probabilities) if y_true.nunique() > 1 else None,
        "rows": len(y_true),
        "positive_labels": int((y_true == 1).sum()),
        "positive_predictions": int((predictions == 1).sum()),
    }


def build_xgb_params(base_scale_pos_weight: float) -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": base_scale_pos_weight,
    }


def create_model(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def run_experiment(
    experiment_name: str,
    feature_set: str,
    feature_columns: list[str],
    split_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    params: dict[str, Any],
    dropped_features: list[str] | None = None,
) -> tuple[list[dict[str, Any]], XGBClassifier, float]:
    x_train, y_train = split_data["train"]
    x_valid, y_valid = split_data["valid"]
    x_test, y_test = split_data["test"]

    model = create_model(params)
    model.fit(x_train[feature_columns], y_train)

    valid_probabilities = get_positive_proba(model, x_valid[feature_columns])
    threshold, _ = find_best_threshold(y_valid, valid_probabilities)
    test_probabilities = get_positive_proba(model, x_test[feature_columns])

    rows: list[dict[str, Any]] = []
    for dataset_name, labels, probabilities in [
        ("validation", y_valid, valid_probabilities),
        ("test", y_test, test_probabilities),
    ]:
        metrics = evaluate_probabilities(labels, probabilities, threshold)
        rows.append(
            {
                "experiment_name": experiment_name,
                "feature_set": feature_set,
                "dataset": dataset_name,
                "feature_count": len(feature_columns),
                "dropped_feature_count": len(dropped_features or []),
                "dropped_features": ",".join(dropped_features or []),
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                **params,
                **metrics,
            }
        )
    return rows, model, threshold


def build_first_grid(base_scale_pos_weight: float) -> list[dict[str, Any]]:
    params_list: list[dict[str, Any]] = []
    for max_depth in [3, 4, 5]:
        for learning_rate in [0.03, 0.05]:
            for n_estimators in [200, 400]:
                params = build_xgb_params(base_scale_pos_weight)
                params.update(
                    {
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                    }
                )
                params_list.append(params)
    return params_list


def select_validation_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df[metrics_df["dataset"] == "validation"].copy()


def sort_by_validation_quality(validation_df: pd.DataFrame) -> pd.DataFrame:
    return validation_df.sort_values(
        by=["f1", "pr_auc", "recall", "precision"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )


def build_high_missing_feature_set(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.60,
) -> tuple[list[str], list[str]]:
    dropped: set[str] = set()
    for column in feature_columns:
        if column.endswith("_missing") or column in PERIOD_FEATURE_COLUMNS:
            continue
        missing_flag = f"{column}_missing"
        if missing_flag in dataframe.columns:
            missing_rate = pd.to_numeric(dataframe[missing_flag], errors="coerce").fillna(0).mean()
        else:
            missing_rate = pd.to_numeric(dataframe[column], errors="coerce").isna().mean()
        if missing_rate >= threshold:
            dropped.add(column)
            if missing_flag in feature_columns:
                dropped.add(missing_flag)
    kept = [column for column in feature_columns if column not in dropped]
    return kept, sorted(dropped)


def build_low_importance_feature_set(
    model: XGBClassifier,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    candidate_importances = importances.drop(labels=list(PERIOD_FEATURE_COLUMNS & set(feature_columns)), errors="ignore")
    if candidate_importances.empty:
        return feature_columns, []

    cutoff = candidate_importances.quantile(0.20)
    dropped = set(candidate_importances[candidate_importances <= cutoff].index.tolist())
    dropped.update(candidate_importances[candidate_importances == 0].index.tolist())
    kept = [column for column in feature_columns if column not in dropped or column in PERIOD_FEATURE_COLUMNS]
    if len(kept) < 5:
        return feature_columns, []
    return kept, sorted(dropped)


def summarize_split(
    split_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    split_groups: dict[str, pd.Series],
) -> pd.DataFrame:
    overlap_counts = validate_group_split(split_groups)
    rows: list[dict[str, Any]] = []
    for split_name, (_, labels) in split_data.items():
        rows.append(
            {
                "split": split_name,
                "rows": len(labels),
                "positive_labels": int((labels == 1).sum()),
                "negative_labels": int((labels == 0).sum()),
                "unique_groups": int(split_groups[split_name].nunique()),
            }
        )
    rows.append({"split": "overlap_train_valid", "rows": overlap_counts["train_valid"]})
    rows.append({"split": "overlap_train_test", "rows": overlap_counts["train_test"]})
    rows.append({"split": "overlap_valid_test", "rows": overlap_counts["valid_test"]})
    return pd.DataFrame(rows)


def markdown_table(dataframe: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if dataframe.empty:
        return "_결과 없음_"
    display = dataframe[columns].head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, separator, *body])


def write_report(
    report_path: Path,
    data_path: Path,
    dataframe: pd.DataFrame,
    split_summary: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_validation_row: pd.Series,
    best_test_row: pd.Series,
    output_paths: dict[str, Path],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_rank = sort_by_validation_quality(select_validation_rows(metrics_df))
    report = f"""# raw 데이터 기반 Group Split XGBoost 학습 결과

## 1. 실험 개요
이번 실험은 기존 전처리 완료 CSV가 아니라 `{data_path}`에서 생성된 raw 기반 학습 CSV를 사용하였다. `종목코드` 기준 Group Split을 적용하여 같은 기업의 데이터가 train, validation, test에 동시에 들어가지 않도록 했고, validation F1을 기준으로 best setting을 선택하였다. PR-AUC는 상폐기업(label=1)처럼 불균형한 양성 클래스를 얼마나 잘 구분하는지 보기 위한 보조 지표로 함께 기록하였다.

## 2. 데이터 변환 결과
- 전체 학습 행 수: {len(dataframe)}
- 고유 종목코드 수: {dataframe["종목코드"].nunique()}
- label 분포: {dataframe["label"].value_counts().sort_index().to_dict()}
- 보고기간 분포: {dataframe["보고기간"].value_counts().sort_index().to_dict()}
- 산업군 분포: {dataframe["산업군"].value_counts().sort_index().to_dict()}

## 3. Group Split 검증
{markdown_table(split_summary, ["split", "rows", "positive_labels", "negative_labels", "unique_groups"], max_rows=6)}

## 4. Validation 기준 상위 결과
{markdown_table(validation_rank, ["experiment_name", "feature_set", "feature_count", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc"], max_rows=10)}

## 5. 최종 선택 결과
최종 best setting은 validation F1 기준 `{best_validation_row["experiment_name"]}`이다. 이 설정은 `{best_validation_row["feature_set"]}` feature set을 사용했고, threshold는 `{best_validation_row["threshold"]:.4f}`로 선택되었다.

- Validation F1: {best_validation_row["f1"]:.4f}
- Validation PR-AUC: {best_validation_row["pr_auc"]:.4f}
- Test F1: {best_test_row["f1"]:.4f}
- Test ROC-AUC: {best_test_row["roc_auc"]:.4f}
- Test PR-AUC: {best_test_row["pr_auc"]:.4f}
- Test precision / recall: {best_test_row["precision"]:.4f} / {best_test_row["recall"]:.4f}

## 6. 해석
이번 실험은 test set에 맞춰 threshold나 feature를 조정하지 않고, validation set에서 선택한 설정을 test set에 그대로 적용하였다. 따라서 test F1이 validation F1보다 낮게 나오더라도 이는 Group Split 기준에서 더 엄격하게 일반화 성능을 확인한 결과로 해석할 수 있다. F1 score 개선은 threshold 최적화와 XGBoost 파라미터 조정을 중심으로 시도했고, feature 제거는 결측이 큰 변수와 importance가 낮은 변수를 줄였을 때 성능이 유지되거나 개선되는지 확인하는 목적이었다.

## 7. 저장 파일
- 실험 metrics: `{output_paths["metrics"]}`
- split summary: `{output_paths["split_summary"]}`
- best model: `{output_paths["best_model"]}`
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = resolve_path(args.data_path) if args.data_path is not None else find_latest_raw_training_csv()
    report_path = resolve_path(args.report_path)

    x_data, y_data, feature_columns, groups, raw_dataframe = load_training_dataframe(data_path)
    split_data, split_groups = split_group_dataset(
        x_data=x_data,
        y_data=y_data,
        groups=groups,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )
    split_summary = summarize_split(split_data, split_groups)

    y_train = split_data["train"][1]
    base_scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    all_metrics: list[dict[str, Any]] = []
    trained_models: dict[str, XGBClassifier] = {}
    thresholds: dict[str, float] = {}
    feature_sets_by_experiment: dict[str, list[str]] = {}

    baseline_params = build_xgb_params(base_scale_pos_weight)
    rows, model, threshold = run_experiment(
        "baseline_default",
        "all_features",
        feature_columns,
        split_data,
        baseline_params,
    )
    all_metrics.extend(rows)
    trained_models["baseline_default"] = model
    thresholds["baseline_default"] = threshold
    feature_sets_by_experiment["baseline_default"] = feature_columns

    first_grid_metrics: list[dict[str, Any]] = []
    for index, params in enumerate(build_first_grid(base_scale_pos_weight), start=1):
        experiment_name = f"grid1_{index:02d}"
        rows, model, threshold = run_experiment(experiment_name, "all_features", feature_columns, split_data, params)
        all_metrics.extend(rows)
        first_grid_metrics.extend(rows)
        trained_models[experiment_name] = model
        thresholds[experiment_name] = threshold
        feature_sets_by_experiment[experiment_name] = feature_columns

    first_grid_df = pd.DataFrame(first_grid_metrics)
    top_first_grid = sort_by_validation_quality(select_validation_rows(first_grid_df)).head(3)
    for _, top_row in top_first_grid.iterrows():
        base_params = json.loads(top_row["params_json"])
        for multiplier in [0.75, 1.0, 1.25]:
            params = dict(base_params)
            params["scale_pos_weight"] = base_scale_pos_weight * multiplier
            experiment_name = f"scale_{top_row['experiment_name']}_{multiplier:.2f}x"
            rows, model, threshold = run_experiment(experiment_name, "all_features", feature_columns, split_data, params)
            all_metrics.extend(rows)
            trained_models[experiment_name] = model
            thresholds[experiment_name] = threshold
            feature_sets_by_experiment[experiment_name] = feature_columns

    metrics_df = pd.DataFrame(all_metrics)
    best_all_validation = sort_by_validation_quality(select_validation_rows(metrics_df)).iloc[0]
    best_all_experiment = str(best_all_validation["experiment_name"])
    best_params = json.loads(best_all_validation["params_json"])
    best_all_model = trained_models[best_all_experiment]

    high_missing_features, high_missing_dropped = build_high_missing_feature_set(raw_dataframe, feature_columns)
    rows, model, threshold = run_experiment(
        "feature_drop_high_missing",
        "drop_high_missing",
        high_missing_features,
        split_data,
        best_params,
        high_missing_dropped,
    )
    all_metrics.extend(rows)
    trained_models["feature_drop_high_missing"] = model
    thresholds["feature_drop_high_missing"] = threshold
    feature_sets_by_experiment["feature_drop_high_missing"] = high_missing_features

    low_importance_features, low_importance_dropped = build_low_importance_feature_set(best_all_model, feature_columns)
    rows, model, threshold = run_experiment(
        "feature_drop_low_importance",
        "drop_low_importance",
        low_importance_features,
        split_data,
        best_params,
        low_importance_dropped,
    )
    all_metrics.extend(rows)
    trained_models["feature_drop_low_importance"] = model
    thresholds["feature_drop_low_importance"] = threshold
    feature_sets_by_experiment["feature_drop_low_importance"] = low_importance_features

    metrics_df = pd.DataFrame(all_metrics)
    validation_rank = sort_by_validation_quality(select_validation_rows(metrics_df))
    best_validation_row = validation_rank.iloc[0]
    best_experiment = str(best_validation_row["experiment_name"])
    best_test_row = metrics_df[
        (metrics_df["experiment_name"] == best_experiment) & (metrics_df["dataset"] == "test")
    ].iloc[0]

    metrics_path = MODEL_DIR / f"raw_xgboost_metrics_{timestamp}.csv"
    split_summary_path = MODEL_DIR / f"raw_xgboost_split_summary_{timestamp}.csv"
    best_model_path = MODEL_DIR / f"raw_xgboost_best_model_{timestamp}.joblib"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    split_summary.to_csv(split_summary_path, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "model": trained_models[best_experiment],
            "experiment_name": best_experiment,
            "threshold": thresholds[best_experiment],
            "feature_columns": feature_sets_by_experiment[best_experiment],
            "data_path": str(data_path),
            "best_validation": best_validation_row.to_dict(),
            "best_test": best_test_row.to_dict(),
        },
        best_model_path,
    )
    write_report(
        report_path=report_path,
        data_path=data_path,
        dataframe=raw_dataframe,
        split_summary=split_summary,
        metrics_df=metrics_df,
        best_validation_row=best_validation_row,
        best_test_row=best_test_row,
        output_paths={
            "metrics": metrics_path,
            "split_summary": split_summary_path,
            "best_model": best_model_path,
        },
    )

    print(f"data path: {data_path}")
    print(f"metrics saved: {metrics_path}")
    print(f"split summary saved: {split_summary_path}")
    print(f"best model saved: {best_model_path}")
    print(f"report saved: {report_path}")
    print(
        "best validation: "
        f"{best_experiment}, f1={best_validation_row['f1']:.4f}, pr_auc={best_validation_row['pr_auc']:.4f}"
    )
    print(
        "best test: "
        f"f1={best_test_row['f1']:.4f}, roc_auc={best_test_row['roc_auc']:.4f}, pr_auc={best_test_row['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()

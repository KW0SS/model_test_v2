from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "재무비율" / "모델학습전처리완료_20260403_001717.csv"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
BEST_MODEL_METRIC = "f1"
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)

METADATA_COLUMNS = ["기업상태", "기업명", "기업코드", "연도", "종목코드"]
DEFAULT_FEATURE_COLUMNS = [
    "총자산증가율",
    "유동자산증가율",
    "매출액증가율",
    "순이익증가율",
    "영업이익증가율",
    "매출액순이익률",
    "매출총이익률",
    "자기자본순이익률 (ROE)",
    "매출채권회전율",
    "재고자산회전율",
    "총자본회전율",
    "유형자산회전율",
    "매출원가율",
    "부채비율",
    "유동비율",
    "자기자본비율",
    "당좌비율",
    "비유동자산장기적합률",
    "순운전자본비율",
    "차입금의존도",
    "현금비율",
    "유형자산",
    "무형자산",
    "무형자산상각비",
    "유형자산상각비",
    "감가상각비",
    "총자본영업이익률",
    "총자본순이익률",
    "유보액/납입자본비율",
    "총자본투자효율",
]
LABEL_COLUMN = "label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="재무비율 데이터 분류 모델 학습 스크립트 (threshold tuning)")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="학습에 사용할 CSV 경로")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help="학습 데이터 비율")
    parser.add_argument("--valid-ratio", type=float, default=VALID_RATIO, help="검증 데이터 비율")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="테스트 데이터 비율")
    return parser.parse_args()


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if min(train_ratio, valid_ratio, test_ratio) <= 0:
        raise ValueError("train/validation/test 비율은 모두 0보다 커야 합니다.")
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"train/validation/test 비율 합은 1이어야 합니다. 현재 합계: {ratio_sum:.6f}")


def infer_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    if all(column in dataframe.columns for column in DEFAULT_FEATURE_COLUMNS):
        if any(column.endswith("_missing") for column in dataframe.columns):
            inferred_columns = [
                column
                for column in dataframe.columns
                if column not in METADATA_COLUMNS and column != LABEL_COLUMN
            ]
            if inferred_columns:
                return inferred_columns
        return DEFAULT_FEATURE_COLUMNS

    inferred_columns = [
        column
        for column in dataframe.columns
        if column not in METADATA_COLUMNS and column != LABEL_COLUMN
    ]
    if not inferred_columns:
        raise KeyError("CSV에서 학습에 사용할 feature 컬럼을 찾지 못했습니다.")
    return inferred_columns


def split_feature_columns(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    base_feature_columns = [column for column in feature_columns if not column.endswith("_missing")]
    indicator_feature_columns = [column for column in feature_columns if column.endswith("_missing")]
    return base_feature_columns, indicator_feature_columns


def signed_log1p_transform(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.log1p(np.abs(array))


def build_feature_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    base_feature_columns, indicator_feature_columns = split_feature_columns(feature_columns)

    # If missing-indicator columns already exist, treat the CSV as preprocessed model input.
    if indicator_feature_columns:
        return ColumnTransformer(
            transformers=[("all_features", "passthrough", feature_columns)],
            remainder="drop",
        )

    transformers: list[tuple[str, Any, list[str]]] = []

    if base_feature_columns:
        transformers.append(
            (
                "base_features",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("signed_log1p", FunctionTransformer(signed_log1p_transform, validate=False)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                base_feature_columns,
            )
        )

    if indicator_feature_columns:
        transformers.append(("missing_indicators", "passthrough", indicator_feature_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    dataframe = pd.read_csv(data_path, encoding="utf-8-sig")
    feature_columns = infer_feature_columns(dataframe)
    missing_columns = [column for column in feature_columns + [LABEL_COLUMN] if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing_columns}")

    dataset = dataframe[feature_columns + [LABEL_COLUMN]].copy()
    dataset[feature_columns] = dataset[feature_columns].apply(pd.to_numeric, errors="coerce")
    dataset[LABEL_COLUMN] = pd.to_numeric(dataset[LABEL_COLUMN], errors="coerce")
    dataset = dataset.dropna(subset=[LABEL_COLUMN]).copy()
    dataset[LABEL_COLUMN] = dataset[LABEL_COLUMN].astype(int)
    return dataset[feature_columns], dataset[LABEL_COLUMN], feature_columns


def split_dataset(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    validate_ratios(train_ratio, valid_ratio, test_ratio)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        train_size=train_ratio,
        random_state=RANDOM_STATE,
        stratify=y_data,
    )
    valid_share_in_temp = valid_ratio / (valid_ratio + test_ratio)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        train_size=valid_share_in_temp,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return {
        "train": (x_train, y_train),
        "valid": (x_valid, y_valid),
        "test": (x_test, y_test),
    }


def build_model_specs(feature_columns: list[str], y_train: pd.Series) -> dict[str, Any]:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count
    feature_preprocessor = build_feature_preprocessor(feature_columns)

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(feature_preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(feature_preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", clone(feature_preprocessor)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def get_positive_proba(estimator: Any, x_eval: pd.DataFrame) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Threshold tuning requires predict_proba support.")
    return estimator.predict_proba(x_eval)[:, 1]


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}

    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        candidate = {"precision": precision, "recall": recall, "f1": f1}
        if (
            candidate["f1"] > best_metrics["f1"]
            or (candidate["f1"] == best_metrics["f1"] and candidate["recall"] > best_metrics["recall"])
            or (
                candidate["f1"] == best_metrics["f1"]
                and candidate["recall"] == best_metrics["recall"]
                and candidate["precision"] > best_metrics["precision"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = candidate

    return best_threshold, best_metrics


def evaluate_with_threshold(
    model_name: str,
    estimator: Any,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    dataset_name: str,
    threshold: float,
) -> dict[str, Any]:
    probabilities = get_positive_proba(estimator, x_eval)
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(y_eval, predictions, zero_division=0),
        "recall": recall_score(y_eval, predictions, zero_division=0),
        "f1": f1_score(y_eval, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, probabilities) if y_eval.nunique() > 1 else None,
        "rows": len(y_eval),
        "positive_labels": int((y_eval == 1).sum()),
        "positive_predictions": int((predictions == 1).sum()),
    }
    return metrics


def format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_split_summary(split_data: dict[str, tuple[pd.DataFrame, pd.Series]]) -> None:
    print("=== Split Summary ===")
    for split_name, (_, labels) in split_data.items():
        counts = labels.value_counts().sort_index().to_dict()
        print(f"{split_name}: rows={len(labels)}, label_counts={counts}")
    print()


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    display_columns = [
        "model",
        "dataset",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "rows",
        "positive_labels",
        "positive_predictions",
    ]
    printable_df = metrics_df[display_columns].copy()
    for column in ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        printable_df[column] = printable_df[column].apply(format_metric)

    print("=== Tuned Model Metrics ===")
    print(printable_df.to_string(index=False))
    print()


def select_best_model(validation_metrics: pd.DataFrame) -> str:
    sorted_df = validation_metrics.sort_values(
        by=[BEST_MODEL_METRIC, "recall", "precision"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return str(sorted_df.iloc[0]["model"])


def save_outputs(
    timestamp: str,
    metrics_df: pd.DataFrame,
    best_model_name: str,
    best_estimator: Any,
    best_threshold: float,
    data_path: Path,
    feature_columns: list[str],
) -> tuple[Path, Path]:
    metrics_output_path = MODEL_DIR / f"metrics_threshold_tuned_{timestamp}.csv"
    model_output_path = MODEL_DIR / f"best_model_threshold_tuned_{best_model_name}_{timestamp}.joblib"

    metrics_df.to_csv(metrics_output_path, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "model_name": best_model_name,
            "estimator": best_estimator,
            "threshold": best_threshold,
            "feature_columns": feature_columns,
            "label_column": LABEL_COLUMN,
            "best_metric": BEST_MODEL_METRIC,
            "data_path": str(data_path),
            "saved_at": timestamp,
        },
        model_output_path,
    )
    return metrics_output_path, model_output_path


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    x_data, y_data, feature_columns = load_dataset(args.data_path)
    split_data = split_dataset(
        x_data=x_data,
        y_data=y_data,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    x_train, y_train = split_data["train"]
    x_valid, y_valid = split_data["valid"]
    x_test, y_test = split_data["test"]

    model_specs = build_model_specs(feature_columns, y_train)
    trained_models: dict[str, Any] = {}
    best_thresholds: dict[str, float] = {}
    metrics_rows: list[dict[str, Any]] = []

    print(f"Data path: {args.data_path}")
    print(f"Feature count: {len(feature_columns)}")
    base_feature_columns, indicator_feature_columns = split_feature_columns(feature_columns)
    print(f"Base feature count: {len(base_feature_columns)}")
    print(f"Missing indicator count: {len(indicator_feature_columns)}")
    print(
        "Preprocessing mode: external preprocessed input"
        if indicator_feature_columns
        else "Preprocessing mode: internal median -> signed log1p -> RobustScaler"
    )
    print_split_summary(split_data)

    for model_name, estimator in model_specs.items():
        model = clone(estimator)
        model.fit(x_train, y_train)
        trained_models[model_name] = model

        valid_probabilities = get_positive_proba(model, x_valid)
        best_threshold, _ = find_best_threshold(y_valid, valid_probabilities)
        best_thresholds[model_name] = best_threshold

        metrics_rows.append(
            evaluate_with_threshold(model_name, model, x_valid, y_valid, "validation", best_threshold)
        )
        metrics_rows.append(
            evaluate_with_threshold(model_name, model, x_test, y_test, "test", best_threshold)
        )

    metrics_df = pd.DataFrame(metrics_rows)
    print_metrics_table(metrics_df)

    validation_metrics = metrics_df[metrics_df["dataset"] == "validation"].copy()
    best_model_name = select_best_model(validation_metrics)
    best_estimator = trained_models[best_model_name]
    best_threshold = best_thresholds[best_model_name]

    metrics_output_path, model_output_path = save_outputs(
        timestamp=timestamp,
        metrics_df=metrics_df,
        best_model_name=best_model_name,
        best_estimator=best_estimator,
        best_threshold=best_threshold,
        data_path=args.data_path,
        feature_columns=feature_columns,
    )

    print(f"Best model ({BEST_MODEL_METRIC} 기준): {best_model_name}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Best model saved to: {model_output_path}")


if __name__ == "__main__":
    main()

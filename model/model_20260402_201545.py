from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "재무비율" / "재무비율_20260402_182020.csv"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
BEST_MODEL_METRIC = "f1"

FEATURE_COLUMNS = [
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
    parser = argparse.ArgumentParser(description="재무비율 데이터 분류 모델 학습 스크립트")
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
        raise ValueError(
            f"train/validation/test 비율 합은 1이어야 합니다. 현재 합계: {ratio_sum:.6f}"
        )


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataframe = pd.read_csv(data_path, encoding="utf-8-sig")

    missing_columns = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN] if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing_columns}")

    dataset = dataframe[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    dataset[FEATURE_COLUMNS] = dataset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dataset[LABEL_COLUMN] = pd.to_numeric(dataset[LABEL_COLUMN], errors="coerce")
    dataset = dataset.dropna(subset=[LABEL_COLUMN]).copy()
    dataset[LABEL_COLUMN] = dataset[LABEL_COLUMN].astype(int)

    x_data = dataset[FEATURE_COLUMNS]
    y_data = dataset[LABEL_COLUMN]
    return x_data, y_data


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


def build_model_specs(y_train: pd.Series) -> dict[str, Any]:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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
                ("imputer", SimpleImputer(strategy="median")),
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
                ("imputer", SimpleImputer(strategy="median")),
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


def evaluate_model(model_name: str, estimator: Any, x_eval: pd.DataFrame, y_eval: pd.Series, dataset_name: str) -> dict[str, Any]:
    predictions = estimator.predict(x_eval)
    metrics: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(y_eval, predictions, zero_division=0),
        "recall": recall_score(y_eval, predictions, zero_division=0),
        "f1": f1_score(y_eval, predictions, zero_division=0),
        "roc_auc": None,
        "rows": len(y_eval),
        "positive_labels": int((y_eval == 1).sum()),
        "positive_predictions": int((predictions == 1).sum()),
    }

    if hasattr(estimator, "predict_proba") and y_eval.nunique() > 1:
        probabilities = estimator.predict_proba(x_eval)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_eval, probabilities)

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
    for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        printable_df[column] = printable_df[column].apply(format_metric)

    print("=== Model Metrics ===")
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
    data_path: Path,
) -> tuple[Path, Path]:
    metrics_output_path = MODEL_DIR / f"metrics_{timestamp}.csv"
    model_output_path = MODEL_DIR / f"best_model_{best_model_name}_{timestamp}.joblib"

    metrics_df.to_csv(metrics_output_path, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "model_name": best_model_name,
            "estimator": best_estimator,
            "feature_columns": FEATURE_COLUMNS,
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

    x_data, y_data = load_dataset(args.data_path)
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

    model_specs = build_model_specs(y_train)
    trained_models: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []

    print(f"Data path: {args.data_path}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print_split_summary(split_data)

    for model_name, estimator in model_specs.items():
        model = clone(estimator)
        model.fit(x_train, y_train)
        trained_models[model_name] = model

        metrics_rows.append(evaluate_model(model_name, model, x_valid, y_valid, "validation"))
        metrics_rows.append(evaluate_model(model_name, model, x_test, y_test, "test"))

    metrics_df = pd.DataFrame(metrics_rows)
    print_metrics_table(metrics_df)

    validation_metrics = metrics_df[metrics_df["dataset"] == "validation"].copy()
    best_model_name = select_best_model(validation_metrics)
    best_estimator = trained_models[best_model_name]

    metrics_output_path, model_output_path = save_outputs(
        timestamp=timestamp,
        metrics_df=metrics_df,
        best_model_name=best_model_name,
        best_estimator=best_estimator,
        data_path=args.data_path,
    )

    print(f"Best model ({BEST_MODEL_METRIC} 기준): {best_model_name}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Best model saved to: {model_output_path}")


if __name__ == "__main__":
    main()

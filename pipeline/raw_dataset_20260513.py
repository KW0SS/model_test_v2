from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import RobustScaler

from pipeline_20260402_180404 import (
    CLIP_RULES,
    MODEL_FEATURE_COLUMNS,
    MODEL_MISSING_FLAG_COLUMNS,
    OUTPUT_DIR,
    RATIO_COLUMNS,
    apply_clip_rule,
    build_model_base_row,
    build_output_row,
    extract_company_fields,
    row_has_meaningful_ratio,
    signed_log1p_series,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "raw"
PERIODS = ["Q1", "H1", "Q3", "ANNUAL"]
PERIOD_FEATURE_COLUMNS = [f"period_{period}" for period in PERIODS]
RAW_METADATA_COLUMNS = ["기업상태", "기업명", "기업코드", "연도", "종목코드", "보고기간", "산업군"]
RAW_FEATURE_COLUMNS = [*MODEL_FEATURE_COLUMNS, *PERIOD_FEATURE_COLUMNS]
RAW_MISSING_FLAG_COLUMNS = [f"{column}_missing" for column in MODEL_FEATURE_COLUMNS]
RAW_OUTPUT_COLUMNS = [*RAW_METADATA_COLUMNS, *RAW_FEATURE_COLUMNS, *RAW_MISSING_FLAG_COLUMNS, "label"]
FILENAME_PATTERN = re.compile(r"^(?P<stock_code>\d{6})_(?P<year>\d{4})_(?P<period>Q1|H1|Q3|ANNUAL)\.json$")


@dataclass(frozen=True)
class RawRecord:
    state_dir: str
    industry: str
    stock_code: str
    year: int
    period: str
    path: Path
    label: int
    company_state: str
    company_name: str
    corp_code: str
    fields: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="raw JSON 기반 모델 학습 CSV 생성")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="raw JSON 루트 폴더")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="CSV 저장 폴더")
    parser.add_argument("--timestamp", type=str, default=None, help="출력 파일명에 사용할 timestamp")
    return parser.parse_args()


def normalize_stock_code(value: str) -> str:
    return str(value or "").strip().zfill(6)


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("financial_statements") or payload.get("list") or []
        return rows if isinstance(rows, list) else []
    return []


def build_raw_record(path: Path, raw_dir: Path) -> tuple[RawRecord | None, str | None]:
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        return None, "invalid_filename"

    try:
        relative = path.relative_to(raw_dir)
        state_dir = relative.parts[0]
        industry = relative.parts[1]
    except (IndexError, ValueError):
        return None, "invalid_path"

    if state_dir not in {"healthy", "delisted"}:
        return None, "invalid_state_dir"

    try:
        rows = load_json_rows(path)
    except (OSError, json.JSONDecodeError):
        return None, "json_load_error"
    if not rows:
        return None, "empty_financial_rows"

    stock_code = normalize_stock_code(match.group("stock_code"))
    year = int(match.group("year"))
    period = match.group("period")
    corp_code = str(rows[0].get("corp_code") or "").strip()
    label = 1 if state_dir == "delisted" else 0
    company_state = "상폐" if label == 1 else "상장"
    fields = extract_company_fields(rows)
    return (
        RawRecord(
            state_dir=state_dir,
            industry=industry,
            stock_code=stock_code,
            year=year,
            period=period,
            path=path,
            label=label,
            company_state=company_state,
            company_name=stock_code,
            corp_code=corp_code,
            fields=fields,
        ),
        None,
    )


def collect_raw_records(raw_dir: Path) -> tuple[list[RawRecord], list[dict[str, str]]]:
    records: list[RawRecord] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(raw_dir.rglob("*.json")):
        record, reason = build_raw_record(path, raw_dir)
        if record is None:
            skipped.append({"path": str(path), "reason": reason or "unknown"})
            continue
        records.append(record)
    return records, skipped


def build_model_rows(records: list[RawRecord]) -> list[dict[str, str]]:
    previous_map = {(record.stock_code, record.period, record.year): record.fields for record in records}
    rows: list[dict[str, str]] = []

    for record in sorted(records, key=lambda item: (item.stock_code, item.period, item.year)):
        raw_row = build_output_row(
            company_state=record.company_state,
            company_name=record.company_name,
            corp_code=record.corp_code,
            stock_code=record.stock_code,
            year=record.year,
            current_fields=record.fields,
            previous_fields=previous_map.get((record.stock_code, record.period, record.year - 1)),
            label=record.label,
        )
        if not row_has_meaningful_ratio(raw_row):
            continue

        model_row = build_model_base_row(raw_row, record.fields)
        model_row["보고기간"] = record.period
        model_row["산업군"] = record.industry
        for period in PERIODS:
            model_row[f"period_{period}"] = "1" if record.period == period else "0"
        rows.append(model_row)

    return rows


def save_raw_datasets(
    model_rows: list[dict[str, str]],
    skipped_rows: list[dict[str, str]],
    output_dir: Path,
    timestamp: str,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"raw_모델학습용_{timestamp}.csv"
    preprocessed_path = output_dir / f"raw_모델학습전처리완료_{timestamp}.csv"
    skipped_path = output_dir / f"raw_skipped_{timestamp}.csv"
    summary_path = output_dir / f"raw_dataset_summary_{timestamp}.csv"

    dataframe = pd.DataFrame(model_rows)
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=RAW_OUTPUT_COLUMNS)
    else:
        for column in MODEL_FEATURE_COLUMNS:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
            dataframe[f"{column}_missing"] = dataframe[column].isna().astype(int)
            dataframe[column] = apply_clip_rule(dataframe[column], CLIP_RULES[column])
        for column in PERIOD_FEATURE_COLUMNS:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0).astype(int)
        dataframe = dataframe[RAW_OUTPUT_COLUMNS].copy()

    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    preprocessed = dataframe.copy()
    if not preprocessed.empty:
        for column in MODEL_FEATURE_COLUMNS:
            median_value = preprocessed[column].median(skipna=True)
            if pd.isna(median_value):
                median_value = 0.0
            preprocessed[column] = preprocessed[column].fillna(median_value)
            preprocessed[column] = signed_log1p_series(preprocessed[column].astype(float))
        scaler = RobustScaler()
        preprocessed[MODEL_FEATURE_COLUMNS] = scaler.fit_transform(preprocessed[MODEL_FEATURE_COLUMNS])
    preprocessed.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False, encoding="utf-8-sig")
    save_summary(dataframe, skipped_rows, summary_path)
    return output_path, preprocessed_path, skipped_path, summary_path


def save_summary(dataframe: pd.DataFrame, skipped_rows: list[dict[str, str]], summary_path: Path) -> None:
    summary_rows: list[dict[str, Any]] = [
        {"metric": "rows", "value": int(len(dataframe))},
        {"metric": "unique_stock_codes", "value": int(dataframe["종목코드"].nunique()) if "종목코드" in dataframe else 0},
        {"metric": "skipped_files", "value": int(len(skipped_rows))},
    ]
    if not dataframe.empty:
        for label, count in dataframe["label"].value_counts(dropna=False).sort_index().items():
            summary_rows.append({"metric": f"label_count::{label}", "value": int(count)})
        for period, count in dataframe["보고기간"].value_counts(dropna=False).sort_index().items():
            summary_rows.append({"metric": f"period_count::{period}", "value": int(count)})
        for state, count in dataframe["기업상태"].value_counts(dropna=False).sort_index().items():
            summary_rows.append({"metric": f"state_count::{state}", "value": int(count)})
        for industry, count in dataframe["산업군"].value_counts(dropna=False).sort_index().items():
            summary_rows.append({"metric": f"industry_count::{industry}", "value": int(count)})
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else (ROOT_DIR / args.raw_dir).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT_DIR / args.output_dir).resolve()

    records, skipped_rows = collect_raw_records(raw_dir)
    model_rows = build_model_rows(records)
    output_path, preprocessed_path, skipped_path, summary_path = save_raw_datasets(
        model_rows=model_rows,
        skipped_rows=skipped_rows,
        output_dir=output_dir,
        timestamp=timestamp,
    )

    print(f"raw files loaded: {len(records)}")
    print(f"model rows: {len(model_rows)}")
    print(f"skipped files: {len(skipped_rows)}")
    print(f"raw training CSV: {output_path}")
    print(f"raw preprocessed CSV: {preprocessed_path}")
    print(f"skipped summary CSV: {skipped_path}")
    print(f"dataset summary CSV: {summary_path}")


if __name__ == "__main__":
    main()

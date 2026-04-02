from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


getcontext().prec = 40

ROOT_DIR = Path(__file__).resolve().parents[1]
FIRST_DATA_DIR = ROOT_DIR / "First_Data"
LISTED_DIR = FIRST_DATA_DIR / "상장기업"
DELISTED_DIR = FIRST_DATA_DIR / "상폐기업"
DELISTED_CSV_PATH = FIRST_DATA_DIR / "2015_2025_delisted.csv"
OUTPUT_DIR = ROOT_DIR / "재무비율"

FILTER_KEYWORDS = [
    "합병",
    "완전자회사 편입 (합병)",
    "공개매수 후 자발적 상장폐지",
    "완전자회사 편입에 따른 상장폐지",
    "주식의 포괄적 교환/이전",
]

RATIO_COLUMNS = [
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

OUTPUT_COLUMNS = [
    "기업상태",
    "기업명",
    "기업코드",
    "연도",
    "종목코드",
    *RATIO_COLUMNS,
    "label",
]

EDA_CORE_RATIOS = [
    "총자산증가율",
    "매출액증가율",
    "순이익증가율",
    "매출액순이익률",
    "자기자본순이익률 (ROE)",
    "부채비율",
    "유동비율",
    "총자본영업이익률",
]

EDA_SCATTER_PAIRS = [
    ("자기자본순이익률 (ROE)", "부채비율"),
    ("매출액순이익률", "유동비율"),
    ("총자본영업이익률", "차입금의존도"),
    ("총자본순이익률", "자기자본비율"),
]

ZERO_DEFAULT_FIELDS = {
    "capital_surplus",
    "shortterm_borrowings",
    "current_portion_of_longterm_borrowings",
    "longterm_borrowings",
    "bonds",
    "interest_expense",
    "intangible_amortization",
    "tangible_depreciation",
}


@dataclass
class DelistedRecord:
    company_name: str
    stock_code: str
    delisting_year: int
    delisting_date: str
    reason: str


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return "".join(str(value).lower().split())


def parse_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))

    text = str(value).strip()
    if not text or text in {"-", "nan", "None"}:
        return None
    text = text.replace(",", "").replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def format_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    quantized = value.quantize(Decimal("0.0000000001"))
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def safe_divide(numerator: Decimal | None, denominator: Decimal | None, multiplier: Decimal = Decimal("1")) -> Decimal | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator) * multiplier


def growth_rate(current: Decimal | None, previous: Decimal | None) -> Decimal | None:
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * Decimal("100")


def text_contains_any(value: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(value)
    return any(normalize_text(keyword) in normalized for keyword in keywords)


def find_first_value(rows: list[dict], *, exact_ids: Iterable[str] = (), id_contains: Iterable[str] = (), name_keywords: Iterable[str] = ()) -> Decimal | None:
    normalized_exact_ids = {normalize_text(item) for item in exact_ids}
    normalized_id_contains = [normalize_text(item) for item in id_contains]
    normalized_name_keywords = [normalize_text(item) for item in name_keywords]

    for row in rows:
        account_id = normalize_text(row.get("account_id"))
        if account_id in normalized_exact_ids:
            amount = parse_decimal(row.get("thstrm_amount"))
            if amount is not None:
                return amount

    for row in rows:
        account_id = normalize_text(row.get("account_id"))
        if any(keyword in account_id for keyword in normalized_id_contains):
            amount = parse_decimal(row.get("thstrm_amount"))
            if amount is not None:
                return amount

    for row in rows:
        account_name = normalize_text(row.get("account_nm"))
        if any(keyword in account_name for keyword in normalized_name_keywords):
            amount = parse_decimal(row.get("thstrm_amount"))
            if amount is not None:
                return amount

    return None


def sum_matching_values(rows: list[dict], *, exact_ids: Iterable[str] = (), id_contains: Iterable[str] = (), name_keywords: Iterable[str] = ()) -> Decimal | None:
    normalized_exact_ids = {normalize_text(item) for item in exact_ids}
    normalized_id_contains = [normalize_text(item) for item in id_contains]
    normalized_name_keywords = [normalize_text(item) for item in name_keywords]

    total = Decimal("0")
    matched = False
    for row in rows:
        account_id = normalize_text(row.get("account_id"))
        account_name = normalize_text(row.get("account_nm"))
        is_match = False
        if account_id in normalized_exact_ids:
            is_match = True
        elif any(keyword in account_id for keyword in normalized_id_contains):
            is_match = True
        elif any(keyword in account_name for keyword in normalized_name_keywords):
            is_match = True

        if is_match:
            amount = parse_decimal(row.get("thstrm_amount"))
            if amount is not None:
                total += amount
                matched = True

    return total if matched else None


def extract_company_fields(financial_rows: list[dict]) -> dict[str, Decimal | None]:
    bs_rows = [row for row in financial_rows if row.get("sj_div") == "BS"]
    cis_rows = [row for row in financial_rows if row.get("sj_div") in {"CIS", "IS"}]
    cf_rows = [row for row in financial_rows if row.get("sj_div") == "CF"]

    fields: dict[str, Decimal | None] = {}

    fields["assets"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_Assets"],
        name_keywords=["자산총계", "총자산"],
    )
    fields["current_assets"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_CurrentAssets"],
        name_keywords=["유동자산"],
    )
    fields["noncurrent_assets"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_NoncurrentAssets"],
        name_keywords=["비유동자산"],
    )
    fields["liabilities"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_Liabilities"],
        name_keywords=["부채총계", "총부채"],
    )
    fields["current_liabilities"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_CurrentLiabilities"],
        name_keywords=["유동부채"],
    )
    fields["equity"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_Equity", "ifrs-full_EquityAttributableToOwnersOfParent"],
        name_keywords=["자본총계", "자기자본"],
    )
    fields["revenue"] = find_first_value(
        cis_rows,
        exact_ids=[
            "ifrs-full_Revenue",
            "ifrs-full_SalesRevenue",
            "ifrs-full_GrossRevenue",
        ],
        name_keywords=["매출액", "영업수익", "수익(매출액)"],
    )
    fields["cost_of_sales"] = find_first_value(
        cis_rows,
        exact_ids=["ifrs-full_CostOfSales"],
        name_keywords=["매출원가"],
    )
    fields["gross_profit"] = find_first_value(
        cis_rows,
        exact_ids=["ifrs-full_GrossProfit"],
        name_keywords=["매출총이익", "매출총손익"],
    )
    fields["operating_income"] = find_first_value(
        cis_rows,
        exact_ids=["dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"],
        name_keywords=["영업이익", "영업손실"],
    )
    fields["profit_loss"] = find_first_value(
        cis_rows,
        exact_ids=[
            "ifrs-full_ProfitLoss",
            "ifrs-full_ProfitLossAttributableToOwnersOfParent",
        ],
        name_keywords=["당기순이익", "당기순손실", "당기순이익(손실)"],
    )
    fields["trade_receivables"] = find_first_value(
        bs_rows,
        exact_ids=[
            "ifrs-full_TradeAndOtherCurrentReceivables",
            "ifrs-full_TradeReceivables",
        ],
        id_contains=["tradereceivable", "tradeandothercurrentreceivable"],
        name_keywords=["매출채권"],
    )
    fields["inventories"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_Inventories"],
        name_keywords=["재고자산"],
    )
    fields["cash_and_cash_equivalents"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_CashAndCashEquivalents"],
        name_keywords=["현금및현금성자산", "현금 및 현금성자산"],
    )
    fields["property_plant_equipment"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_PropertyPlantAndEquipment"],
        name_keywords=["유형자산"],
    )
    fields["intangible_assets"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_IntangibleAssetsOtherThanGoodwill", "ifrs-full_IntangibleAssets"],
        name_keywords=["무형자산"],
    )
    fields["issued_capital"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_IssuedCapital"],
        name_keywords=["자본금", "납입자본금"],
    )
    fields["retained_earnings"] = find_first_value(
        bs_rows,
        exact_ids=["ifrs-full_RetainedEarnings"],
        name_keywords=["이익잉여금", "결손금"],
    )
    fields["capital_surplus"] = find_first_value(
        bs_rows,
        exact_ids=["dart_CapitalSurplus", "ifrs-full_SharePremium"],
        name_keywords=["자본잉여금", "주식발행초과금", "기타불입자본"],
    )
    fields["shortterm_borrowings"] = sum_matching_values(
        bs_rows,
        exact_ids=["ifrs-full_ShorttermBorrowings"],
        id_contains=["shorttermborrowings"],
        name_keywords=["단기차입금"],
    )
    fields["current_portion_of_longterm_borrowings"] = sum_matching_values(
        bs_rows,
        exact_ids=["ifrs-full_CurrentPortionOfLongtermBorrowings"],
        id_contains=["currentportionoflongtermborrowings"],
        name_keywords=["유동성장기부채", "유동성장기차입금", "유동성사채", "유동성장기차입부채"],
    )
    fields["longterm_borrowings"] = sum_matching_values(
        bs_rows,
        exact_ids=["dart_LongTermBorrowingsGross"],
        id_contains=["longtermborrowings"],
        name_keywords=["장기차입금"],
    )
    fields["bonds"] = sum_matching_values(
        bs_rows,
        id_contains=["bond", "debenture"],
        name_keywords=["사채", "전환사채", "신주인수권부사채"],
    )
    fields["interest_expense"] = find_first_value(
        cis_rows,
        exact_ids=["ifrs-full_FinanceCosts"],
        id_contains=["interestexpense"],
        name_keywords=["이자비용", "금융비용"],
    )
    fields["intangible_amortization"] = find_first_value(
        cf_rows,
        exact_ids=["ifrs-full_AdjustmentsForAmortisationExpense"],
        id_contains=["amortisationexpense", "amortizationexpense"],
        name_keywords=["무형자산상각비"],
    )
    fields["tangible_depreciation"] = find_first_value(
        cf_rows,
        exact_ids=["ifrs-full_AdjustmentsForDepreciationExpense"],
        id_contains=["depreciationexpense"],
        name_keywords=["감가상각비", "유형자산상각비"],
    )

    if fields["noncurrent_assets"] is None and fields["assets"] is not None and fields["current_assets"] is not None:
        fields["noncurrent_assets"] = fields["assets"] - fields["current_assets"]
    if fields["liabilities"] is None and fields["assets"] is not None and fields["equity"] is not None:
        fields["liabilities"] = fields["assets"] - fields["equity"]
    if fields["equity"] is None and fields["assets"] is not None and fields["liabilities"] is not None:
        fields["equity"] = fields["assets"] - fields["liabilities"]
    if fields["gross_profit"] is None and fields["revenue"] is not None and fields["cost_of_sales"] is not None:
        fields["gross_profit"] = fields["revenue"] - fields["cost_of_sales"]
    if fields["cost_of_sales"] is None and fields["revenue"] is not None and fields["gross_profit"] is not None:
        fields["cost_of_sales"] = fields["revenue"] - fields["gross_profit"]

    for field_name in ZERO_DEFAULT_FIELDS:
        if fields.get(field_name) is None:
            fields[field_name] = Decimal("0")

    return fields


def load_annual_data(json_path: Path) -> dict | None:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    request = payload.get("request", {})
    company = payload.get("company", {})
    financial_rows = payload.get("financial_statements") or []
    year_value = request.get("bsns_year")
    try:
        year = int(year_value)
    except (TypeError, ValueError):
        return None

    fields = extract_company_fields(financial_rows)
    return {
        "year": year,
        "company_name": company.get("corp_name") or json_path.parent.parent.name,
        "corp_code": str(company.get("corp_code") or ""),
        "stock_code": str(company.get("stock_code") or "").zfill(6),
        "fields": fields,
    }


def get_annual_file(year_dir: Path) -> Path | None:
    candidates = sorted(year_dir.glob("*11011.json"))
    return candidates[0] if candidates else None


def load_company_year_map(company_dir: Path) -> dict[int, dict]:
    year_map: dict[int, dict] = {}
    for year_dir in sorted((path for path in company_dir.iterdir() if path.is_dir()), key=lambda item: item.name):
        annual_file = get_annual_file(year_dir)
        if annual_file is None:
            continue
        annual_data = load_annual_data(annual_file)
        if annual_data is None:
            continue
        year_map[annual_data["year"]] = annual_data
    return year_map


def read_delisted_records() -> tuple[dict[str, DelistedRecord], set[str], int]:
    included: dict[str, DelistedRecord] = {}
    excluded_codes: set[str] = set()
    total_rows = 0

    with DELISTED_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            total_rows += 1
            stock_code = str(row.get("종목코드") or "").strip().zfill(6)
            reason = str(row.get("폐지사유") or "").strip()
            delisting_date = str(row.get("상폐일") or "").strip()

            if not stock_code or len(delisting_date) < 4 or not delisting_date[:4].isdigit():
                continue
            if any(keyword in reason for keyword in FILTER_KEYWORDS):
                excluded_codes.add(stock_code)
                continue

            included[stock_code] = DelistedRecord(
                company_name=str(row.get("기업명") or "").strip(),
                stock_code=stock_code,
                delisting_year=int(delisting_date[:4]),
                delisting_date=delisting_date,
                reason=reason,
            )

    return included, excluded_codes, total_rows


def build_output_row(
    company_state: str,
    company_name: str,
    corp_code: str,
    stock_code: str,
    year: int,
    current_fields: dict[str, Decimal | None],
    previous_fields: dict[str, Decimal | None] | None,
    label: int,
) -> dict[str, str]:
    previous_fields = previous_fields or {}

    total_capital = current_fields.get("assets")
    debt_amount = (
        (current_fields.get("shortterm_borrowings") or Decimal("0"))
        + (current_fields.get("current_portion_of_longterm_borrowings") or Decimal("0"))
        + (current_fields.get("longterm_borrowings") or Decimal("0"))
        + (current_fields.get("bonds") or Decimal("0"))
    )
    intangible_amortization = current_fields.get("intangible_amortization")
    tangible_depreciation = current_fields.get("tangible_depreciation")
    total_depreciation = None
    if intangible_amortization is not None and tangible_depreciation is not None:
        total_depreciation = intangible_amortization + tangible_depreciation

    quick_assets = None
    if current_fields.get("current_assets") is not None and current_fields.get("inventories") is not None:
        quick_assets = current_fields["current_assets"] - current_fields["inventories"]

    net_working_capital = None
    if current_fields.get("current_assets") is not None and current_fields.get("current_liabilities") is not None:
        net_working_capital = current_fields["current_assets"] - current_fields["current_liabilities"]

    reserve_amount = None
    if current_fields.get("retained_earnings") is not None and current_fields.get("capital_surplus") is not None:
        reserve_amount = current_fields["retained_earnings"] + current_fields["capital_surplus"]

    ratio_values: dict[str, Decimal | None] = {
        "총자산증가율": growth_rate(current_fields.get("assets"), previous_fields.get("assets")),
        "유동자산증가율": growth_rate(current_fields.get("current_assets"), previous_fields.get("current_assets")),
        "매출액증가율": growth_rate(current_fields.get("revenue"), previous_fields.get("revenue")),
        "순이익증가율": growth_rate(current_fields.get("profit_loss"), previous_fields.get("profit_loss")),
        "영업이익증가율": growth_rate(current_fields.get("operating_income"), previous_fields.get("operating_income")),
        "매출액순이익률": safe_divide(current_fields.get("profit_loss"), current_fields.get("revenue"), Decimal("100")),
        "매출총이익률": safe_divide(current_fields.get("gross_profit"), current_fields.get("revenue"), Decimal("100")),
        "자기자본순이익률 (ROE)": safe_divide(current_fields.get("profit_loss"), current_fields.get("equity"), Decimal("100")),
        "매출채권회전율": safe_divide(current_fields.get("revenue"), current_fields.get("trade_receivables")),
        "재고자산회전율": safe_divide(current_fields.get("cost_of_sales"), current_fields.get("inventories")),
        "총자본회전율": safe_divide(current_fields.get("revenue"), total_capital),
        "유형자산회전율": safe_divide(current_fields.get("revenue"), current_fields.get("property_plant_equipment")),
        "매출원가율": safe_divide(current_fields.get("cost_of_sales"), current_fields.get("revenue"), Decimal("100")),
        "부채비율": safe_divide(current_fields.get("liabilities"), current_fields.get("equity"), Decimal("100")),
        "유동비율": safe_divide(current_fields.get("current_assets"), current_fields.get("current_liabilities"), Decimal("100")),
        "자기자본비율": safe_divide(current_fields.get("equity"), current_fields.get("assets"), Decimal("100")),
        "당좌비율": safe_divide(quick_assets, current_fields.get("current_liabilities"), Decimal("100")),
        "비유동자산장기적합률": safe_divide(current_fields.get("noncurrent_assets"), current_fields.get("longterm_borrowings")),
        "순운전자본비율": safe_divide(net_working_capital, total_capital, Decimal("100")),
        "차입금의존도": safe_divide(debt_amount, total_capital, Decimal("100")),
        "현금비율": safe_divide(current_fields.get("cash_and_cash_equivalents"), current_fields.get("current_liabilities"), Decimal("100")),
        "유형자산": current_fields.get("property_plant_equipment"),
        "무형자산": current_fields.get("intangible_assets"),
        "무형자산상각비": intangible_amortization,
        "유형자산상각비": tangible_depreciation,
        "감가상각비": total_depreciation,
        "총자본영업이익률": safe_divide(current_fields.get("operating_income"), total_capital, Decimal("100")),
        "총자본순이익률": safe_divide(current_fields.get("profit_loss"), total_capital, Decimal("100")),
        "유보액/납입자본비율": safe_divide(reserve_amount, current_fields.get("issued_capital"), Decimal("100")),
        "총자본투자효율": safe_divide(
            (current_fields.get("profit_loss") or None)
            if current_fields.get("profit_loss") is None and current_fields.get("interest_expense") is None
            else (current_fields.get("profit_loss") or Decimal("0")) + (current_fields.get("interest_expense") or Decimal("0")),
            total_capital,
        ),
    }

    row = {
        "기업상태": company_state,
        "기업명": company_name,
        "기업코드": corp_code,
        "연도": str(year),
        "종목코드": stock_code,
        "label": str(label),
    }
    for ratio_name in RATIO_COLUMNS:
        row[ratio_name] = format_decimal(ratio_values[ratio_name])
    return row


def row_has_meaningful_ratio(row: dict[str, str]) -> bool:
    for column in RATIO_COLUMNS:
        value = str(row.get(column, "")).strip()
        if not value:
            continue

        parsed_value = parse_decimal(value)
        if parsed_value is None:
            return True
        if parsed_value != 0:
            return True

    return False


def iter_company_dirs(base_dir: Path) -> Iterable[Path]:
    return sorted((path for path in base_dir.iterdir() if path.is_dir()), key=lambda item: item.name)


def process_listed_companies(writer: csv.DictWriter) -> int:
    written_rows = 0
    for index, company_dir in enumerate(iter_company_dirs(LISTED_DIR), start=1):
        year_map = load_company_year_map(company_dir)
        if not year_map:
            continue

        for year in sorted(year_map):
            current = year_map[year]
            previous_fields = year_map.get(year - 1, {}).get("fields")
            row = build_output_row(
                company_state="정상기업",
                company_name=current["company_name"],
                corp_code=current["corp_code"],
                stock_code=current["stock_code"],
                year=year,
                current_fields=current["fields"],
                previous_fields=previous_fields,
                label=0,
            )
            if row_has_meaningful_ratio(row):
                writer.writerow(row)
                written_rows += 1

        if index % 100 == 0:
            print(f"[상장기업] {index}개 기업 처리 완료, 현재 행 수: {written_rows}")

    return written_rows


def build_delisted_target_years(delisted_record: DelistedRecord, use_last_year: bool) -> dict[int, int]:
    if use_last_year:
        return {
            delisted_record.delisting_year - 1: 1,
            delisted_record.delisting_year - 2: 0,
            delisted_record.delisting_year - 3: 0,
        }
    return {
        delisted_record.delisting_year - 2: 1,
        delisted_record.delisting_year - 3: 0,
    }


def process_delisted_companies(
    writer: csv.DictWriter,
    included_records: dict[str, DelistedRecord],
    excluded_codes: set[str],
    use_last_year: bool = False,
) -> int:
    written_rows = 0
    matched_companies = 0

    for index, company_dir in enumerate(iter_company_dirs(DELISTED_DIR), start=1):
        year_map = load_company_year_map(company_dir)
        if not year_map:
            continue

        sample_data = next(iter(year_map.values()))
        stock_code = sample_data["stock_code"]
        if stock_code in excluded_codes:
            continue

        delisted_record = included_records.get(stock_code)
        if delisted_record is None:
            continue

        matched_companies += 1
        target_years = build_delisted_target_years(delisted_record, use_last_year)

        for year, label in sorted(target_years.items()):
            current = year_map.get(year)
            if current is None:
                continue
            previous_fields = year_map.get(year - 1, {}).get("fields")
            row = build_output_row(
                company_state="상폐기업",
                company_name=current["company_name"],
                corp_code=current["corp_code"],
                stock_code=current["stock_code"],
                year=year,
                current_fields=current["fields"],
                previous_fields=previous_fields,
                label=label,
            )
            if row_has_meaningful_ratio(row):
                writer.writerow(row)
                written_rows += 1

        if index % 50 == 0:
            print(f"[상폐기업] {index}개 기업 확인 완료, 매칭 기업 수: {matched_companies}, 현재 행 수: {written_rows}")

    print(f"[상폐기업] 최종 매칭 기업 수: {matched_companies}")
    return written_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="재무비율 파이프라인 및 EDA 도구")
    parser.add_argument(
        "--with-eda",
        action="store_true",
        help="재무비율 CSV 생성 후 바로 EDA를 수행합니다.",
    )
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="파이프라인 계산은 건너뛰고 기존 결과 CSV로 EDA만 수행합니다.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="EDA 대상 CSV 경로입니다. --eda-only에서 미지정 시 최신 CSV를 사용합니다.",
    )
    parser.add_argument(
        "--eda-output-dir",
        type=Path,
        help="EDA 결과 저장 폴더입니다. 미지정 시 재무비율/eda_타임스탬프 를 사용합니다.",
    )
    parser.add_argument(
        "--use-last-year",
        action="store_true",
        help="상폐기업 데이터에 상폐연도-1은 label=1, 상폐연도-2/-3은 label=0으로 저장합니다.",
    )
    args = parser.parse_args()
    if args.with_eda and args.eda_only:
        parser.error("--with-eda 와 --eda-only 는 동시에 사용할 수 없습니다.")
    if args.input_csv and not args.eda_only:
        print("[안내] --input-csv 는 --eda-only 에서만 사용됩니다. 현재 값은 무시됩니다.")
    return args


def generate_financial_ratio_csv(use_last_year: bool = False) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    included_records, excluded_codes, total_delisted_rows = read_delisted_records()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"재무비율_{timestamp}.csv"

    print(f"원본 상폐 CSV 행 수: {total_delisted_rows}")
    print(f"필터링으로 제외된 상폐 종목 수: {len(excluded_codes)}")
    print(f"계산 대상으로 남은 상폐 종목 수: {len(included_records)}")
    if use_last_year:
        print("상폐기업 라벨 규칙: 상폐연도-1은 label=1, 상폐연도-2/-3은 label=0")
    else:
        print("상폐기업 라벨 규칙: 상폐연도-2는 label=1, 상폐연도-3은 label=0")
    print(f"결과 저장 경로: {output_path}")

    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        listed_rows = process_listed_companies(writer)
        delisted_rows = process_delisted_companies(
            writer,
            included_records,
            excluded_codes,
            use_last_year=use_last_year,
        )

    print(f"상장기업 행 수: {listed_rows}")
    print(f"상폐기업 행 수: {delisted_rows}")
    print(f"총 저장 행 수: {listed_rows + delisted_rows}")
    return output_path


def find_latest_output_csv() -> Path:
    candidates = sorted(
        OUTPUT_DIR.glob("재무비율_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"EDA 대상 CSV를 찾지 못했습니다: {OUTPUT_DIR}")
    return candidates[0]


def resolve_eda_input_csv(input_csv: Path | None) -> Path:
    if input_csv is None:
        return find_latest_output_csv()
    return input_csv if input_csv.is_absolute() else (ROOT_DIR / input_csv).resolve()


def build_eda_output_dir(eda_output_dir: Path | None) -> Path:
    if eda_output_dir is not None:
        path = eda_output_dir if eda_output_dir.is_absolute() else (ROOT_DIR / eda_output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"eda_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_plot_libraries():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EDA 그래프 생성을 위해 seaborn 이 필요합니다. 먼저 `python -m pip install seaborn` 를 실행해주세요."
        ) from exc

    return plt, font_manager, sns


def configure_plot_font(plt, font_manager, sns) -> None:
    preferred_fonts = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = "DejaVu Sans"
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "font.family": selected_font,
            "font.sans-serif": [selected_font, "Malgun Gothic", "NanumGothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
        },
    )
    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = [selected_font, "Malgun Gothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_eda_dataframe(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    dataframe["연도"] = pd.to_numeric(dataframe["연도"], errors="coerce").astype("Int64")
    dataframe["label"] = pd.to_numeric(dataframe["label"], errors="coerce").astype("Int64")
    for column in RATIO_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def clip_series_for_plot(series: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return valid
    lower = valid.quantile(lower_quantile)
    upper = valid.quantile(upper_quantile)
    return valid.clip(lower=lower, upper=upper)


def save_overview_summary(dataframe: pd.DataFrame, output_dir: Path) -> None:
    summary_rows = [
        {"metric": "total_rows", "value": int(len(dataframe))},
        {"metric": "unique_companies", "value": int(dataframe["종목코드"].nunique())},
        {"metric": "year_min", "value": int(dataframe["연도"].min()) if dataframe["연도"].notna().any() else ""},
        {"metric": "year_max", "value": int(dataframe["연도"].max()) if dataframe["연도"].notna().any() else ""},
    ]
    for state, count in dataframe["기업상태"].value_counts(dropna=False).items():
        summary_rows.append({"metric": f"state_count::{state}", "value": int(count)})
    for label, count in dataframe["label"].value_counts(dropna=False).sort_index().items():
        summary_rows.append({"metric": f"label_count::{label}", "value": int(count)})
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_overview.csv", index=False, encoding="utf-8-sig")


def save_missing_zero_summary(dataframe: pd.DataFrame, output_dir: Path) -> None:
    summary_rows = []
    total_rows = len(dataframe)
    for column in RATIO_COLUMNS:
        series = dataframe[column]
        valid = series.dropna()
        summary_rows.append(
            {
                "ratio_name": column,
                "valid_count": int(valid.shape[0]),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()) if total_rows else 0.0,
                "zero_count": int((valid == 0).sum()),
                "zero_rate": float((valid == 0).mean()) if not valid.empty else 0.0,
                "mean": float(valid.mean()) if not valid.empty else np.nan,
                "median": float(valid.median()) if not valid.empty else np.nan,
                "std": float(valid.std()) if not valid.empty else np.nan,
                "min": float(valid.min()) if not valid.empty else np.nan,
                "max": float(valid.max()) if not valid.empty else np.nan,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_missing_zero.csv", index=False, encoding="utf-8-sig")


def save_by_label_summary(dataframe: pd.DataFrame, output_dir: Path) -> None:
    label_zero = dataframe[dataframe["label"] == 0]
    label_one = dataframe[dataframe["label"] == 1]
    summary_rows = []
    for column in RATIO_COLUMNS:
        series_zero = label_zero[column].dropna()
        series_one = label_one[column].dropna()
        summary_rows.append(
            {
                "ratio_name": column,
                "label_0_count": int(series_zero.shape[0]),
                "label_0_mean": float(series_zero.mean()) if not series_zero.empty else np.nan,
                "label_0_median": float(series_zero.median()) if not series_zero.empty else np.nan,
                "label_1_count": int(series_one.shape[0]),
                "label_1_mean": float(series_one.mean()) if not series_one.empty else np.nan,
                "label_1_median": float(series_one.median()) if not series_one.empty else np.nan,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_by_label.csv", index=False, encoding="utf-8-sig")


def plot_label_distribution(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = dataframe.copy()
    plot_df["label_str"] = plot_df["label"].astype(str)
    sns.countplot(data=plot_df, x="label_str", order=sorted(plot_df["label_str"].dropna().unique()), palette="Blues", ax=ax)
    ax.set_title("Label Distribution")
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    for idx, patch in enumerate(ax.patches):
        value = patch.get_height()
        ax.text(idx, value, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "01_label_distribution.png", dpi=200)
    plt.close(fig)


def plot_state_distribution(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = dataframe["기업상태"].value_counts().index.tolist()
    sns.countplot(data=dataframe, x="기업상태", order=order, palette="Set2", ax=ax)
    ax.set_title("Company State Distribution")
    ax.set_xlabel("기업상태")
    ax.set_ylabel("count")
    for idx, patch in enumerate(ax.patches):
        value = patch.get_height()
        ax.text(idx, value, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "02_state_distribution.png", dpi=200)
    plt.close(fig)


def plot_yearly_state_counts(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = dataframe.dropna(subset=["연도"]).copy()
    plot_df["연도_str"] = plot_df["연도"].astype(int).astype(str)
    sns.histplot(
        data=plot_df,
        x="연도_str",
        hue="기업상태",
        multiple="stack",
        discrete=True,
        shrink=0.85,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Yearly Counts by Company State")
    ax.set_xlabel("연도")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "03_yearly_state_counts.png", dpi=200)
    plt.close(fig)


def plot_yearly_label_counts(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = dataframe.dropna(subset=["연도", "label"]).copy()
    plot_df["연도_str"] = plot_df["연도"].astype(int).astype(str)
    plot_df["label_str"] = plot_df["label"].astype(int).astype(str)
    sns.histplot(
        data=plot_df,
        x="연도_str",
        hue="label_str",
        multiple="stack",
        discrete=True,
        shrink=0.85,
        palette="coolwarm",
        ax=ax,
    )
    ax.set_title("Yearly Counts by Label")
    ax.set_xlabel("연도")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "04_yearly_label_counts.png", dpi=200)
    plt.close(fig)


def plot_missing_zero_rates(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    missing_rates = dataframe[RATIO_COLUMNS].isna().mean().sort_values(ascending=False)
    zero_rates = dataframe[RATIO_COLUMNS].apply(lambda col: (col.dropna() == 0).mean() if col.dropna().shape[0] else 0.0).sort_values(ascending=False)

    missing_df = pd.DataFrame({"ratio_name": missing_rates.index, "rate": missing_rates.values})
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=missing_df, x="rate", y="ratio_name", palette="Blues_r", ax=ax)
    ax.set_title("Missing Rate by Ratio")
    ax.set_xlabel("missing rate")
    fig.tight_layout()
    fig.savefig(output_dir / "05_missing_rate_top30.png", dpi=200)
    plt.close(fig)

    zero_df = pd.DataFrame({"ratio_name": zero_rates.index, "rate": zero_rates.values})
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=zero_df, x="rate", y="ratio_name", palette="Oranges_r", ax=ax)
    ax.set_title("Zero Rate by Ratio")
    ax.set_xlabel("zero rate")
    fig.tight_layout()
    fig.savefig(output_dir / "06_zero_rate_top30.png", dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    correlation = dataframe[RATIO_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(correlation, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "07_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def plot_boxplots_by_label(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()

    for ax, column in zip(axes, EDA_CORE_RATIOS):
        plot_df = dataframe[[column, "label"]].dropna().copy()
        plot_df[column] = clip_series_for_plot(plot_df[column]).reindex(plot_df.index)
        plot_df = plot_df.dropna()
        plot_df["label_str"] = plot_df["label"].astype(int).astype(str)
        sns.boxplot(data=plot_df, x="label_str", y=column, showfliers=False, palette="coolwarm", ax=ax)
        ax.set_title(column)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Core Ratio Boxplots by Label", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "08_boxplots_by_label.png", dpi=200)
    plt.close(fig)


def plot_histograms_core_ratios(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()
    for ax, column in zip(axes, EDA_CORE_RATIOS):
        plot_df = dataframe[[column, "label"]].dropna().copy()
        plot_df[column] = clip_series_for_plot(plot_df[column]).reindex(plot_df.index)
        plot_df = plot_df.dropna()
        plot_df["label_str"] = plot_df["label"].astype(int).astype(str)
        if not plot_df.empty:
            sns.histplot(
                data=plot_df,
                x=column,
                hue="label_str",
                bins=30,
                stat="count",
                common_norm=False,
                element="step",
                fill=True,
                alpha=0.35,
                palette="coolwarm",
                ax=ax,
            )
        ax.set_title(column)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Core Ratio Histograms", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "09_histograms_core_ratios.png", dpi=200)
    plt.close(fig)


def plot_scatter_pairs(dataframe: pd.DataFrame, output_dir: Path, plt, sns) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (x_col, y_col) in zip(axes, EDA_SCATTER_PAIRS):
        plot_df = dataframe[[x_col, y_col, "label"]].dropna().copy()
        if plot_df.empty:
            ax.set_title(f"{x_col} vs {y_col}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        plot_df[x_col] = clip_series_for_plot(plot_df[x_col]).reindex(plot_df.index)
        plot_df[y_col] = clip_series_for_plot(plot_df[y_col]).reindex(plot_df.index)
        plot_df = plot_df.dropna()
        plot_df["label_str"] = plot_df["label"].astype(int).astype(str)
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue="label_str",
            palette="coolwarm",
            s=28,
            alpha=0.4,
            ax=ax,
        )

        ax.set_title(f"{x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(alpha=0.25)

    fig.suptitle("Profitability vs Stability Scatter Plots", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "10_scatter_profitability_vs_stability.png", dpi=200)
    plt.close(fig)


def run_eda(csv_path: Path, eda_output_dir: Path) -> None:
    plt, font_manager, sns = require_plot_libraries()
    configure_plot_font(plt, font_manager, sns)

    print(f"EDA 입력 CSV: {csv_path}")
    print(f"EDA 저장 폴더: {eda_output_dir}")

    dataframe = load_eda_dataframe(csv_path)

    save_overview_summary(dataframe, eda_output_dir)
    save_missing_zero_summary(dataframe, eda_output_dir)
    save_by_label_summary(dataframe, eda_output_dir)

    plot_label_distribution(dataframe, eda_output_dir, plt, sns)
    plot_state_distribution(dataframe, eda_output_dir, plt, sns)
    plot_yearly_state_counts(dataframe, eda_output_dir, plt, sns)
    plot_yearly_label_counts(dataframe, eda_output_dir, plt, sns)
    plot_missing_zero_rates(dataframe, eda_output_dir, plt, sns)
    plot_correlation_heatmap(dataframe, eda_output_dir, plt, sns)
    plot_boxplots_by_label(dataframe, eda_output_dir, plt, sns)
    plot_histograms_core_ratios(dataframe, eda_output_dir, plt, sns)
    plot_scatter_pairs(dataframe, eda_output_dir, plt, sns)

    print("EDA 요약 CSV 3개와 PNG 그래프 10개 저장이 완료되었습니다.")


def main() -> None:
    args = parse_args()

    try:
        if args.eda_only:
            input_csv = resolve_eda_input_csv(args.input_csv)
            eda_output_dir = build_eda_output_dir(args.eda_output_dir)
            run_eda(input_csv, eda_output_dir)
            return

        output_csv = generate_financial_ratio_csv(use_last_year=args.use_last_year)
        if args.with_eda:
            eda_output_dir = build_eda_output_dir(args.eda_output_dir)
            run_eda(output_csv, eda_output_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[오류] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

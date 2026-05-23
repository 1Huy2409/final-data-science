from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from datasets import DownloadConfig, load_dataset
from sklearn.model_selection import train_test_split


DATASET_ID = "tinixai/vietnamese-job-descriptions"
TEXT_COLUMNS = [
    "job_title",
    "company_name",
    "location",
    "job_type",
    "job_industry",
    "experience_level",
    "education_level",
    "job_position",
    "job_description",
    "benefits",
    "requirements",
]
LONG_TEXT_COLUMNS = ["job_description", "benefits", "requirements"]
RAW_TEXT_LENGTH_COLUMNS = ["job_title", "job_description", "benefits", "requirements", "location"]
CATEGORICAL_COLUMNS = [
    "job_type",
    "job_industry",
    "experience_level",
    "education_level",
    "job_position",
    "location",
]
AMBIGUOUS_SALARY_KEYWORDS = [
    "thoa thuan",
    "thương lượng",
    "thuong luong",
    "canh tranh",
    "cạnh tranh",
    "negotiable",
    "deal luong",
    "dang cap nhat",
    "đang cập nhật",
    "updating",
    "update",
    "competitive",
]
SALARY_LEAKAGE_REGEX = re.compile(
    r"(?ix)"
    r"("
    r"(luong|thu\s*nhap|salary)\s*[:\-]?\s*"
    r"[\$]?\s*\d[\d\.,]*\s*(usd|us\$|\$|trieu|tri[eệ]u|tr|vnd|vnđ|dong|/thang|/month)?"
    r"(\s*(\-|~|to|den|t[ơo]i)\s*[\$]?\s*\d[\d\.,]*\s*(usd|us\$|\$|trieu|tri[eệ]u|tr|vnd|vnđ|dong|/thang|/month)?)?"
    r"|"
    r"[\$]?\s*\d[\d\.,]*\s*(usd|us\$|\$|trieu|tri[eệ]u|tr|vnd|vnđ|dong|/thang|/month)"
    r"(\s*(\-|~|to|den|t[ơo]i)\s*[\$]?\s*\d[\d\.,]*\s*(usd|us\$|\$|trieu|tri[eệ]u|tr|vnd|vnđ|dong|/thang|/month))?"
    r")"
)
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
PHONE_REGEX = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s\.-]?\d){8,10}(?!\d)")
HTML_REGEX = re.compile(r"<[^>]+>")
WHITESPACE_REGEX = re.compile(r"\s+")
REPEATED_PUNCT_REGEX = re.compile(r"([!?,.;:\-_=+/\\|])\1+")
NUMBER_TOKEN_REGEX = re.compile(r"\d[\d\.,]*")
RANGE_SPLIT_REGEX = re.compile(r"\s*(?:\-|~|to|den|t[ơo]i|–|—)\s*", flags=re.IGNORECASE)
# Khớp đơn vị triệu VNĐ một cách chính xác dựa trên ranh giới từ (\b).
# Tránh khớp nhầm các từ tiếng Việt chứa chuỗi 'tr' như 'trở lên', 'trách nhiệm', 'trong'.
MILLION_REGEX = re.compile(
    r"\b(?:trieu|triệu)\b|\btr\b|(?<=\d)(?:tr|trieu|triệu)\b",
    re.IGNORECASE,
)


@dataclass
class PipelineConfig:
    data_dir: Path = Path("artifacts")
    dataset_id: str = DATASET_ID
    usd_to_vnd: float = 25000.0
    test_size: float = 0.1
    random_state: int = 42
    sample_size: int | None = None
    remove_salary_mentions: bool = False

    @property
    def usd_to_million_vnd(self) -> float:
        return self.usd_to_vnd / 1_000_000.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audit"

    @property
    def figures_dir(self) -> Path:
        return self.data_dir / "figures"


def ensure_directories(config: PipelineConfig) -> None:
    for path in [config.data_dir, config.raw_dir, config.clean_dir, config.audit_dir, config.figures_dir]:
        path.mkdir(parents=True, exist_ok=True)


def find_cached_dataset_file() -> Path | None:
    home = Path.home()
    parquet_candidates = list(
        home.glob(".cache/huggingface/hub/datasets--tinixai--vietnamese-job-descriptions/snapshots/*/data.parquet")
    )
    if parquet_candidates:
        return parquet_candidates[-1]
    return None


def read_cached_file(file_path: Path, sample_size: int | None = None) -> pd.DataFrame:
    if sample_size:
        parquet_file = pq.ParquetFile(file_path)
        rows_remaining = sample_size
        batches = []
        for batch in parquet_file.iter_batches(batch_size=min(sample_size, 50_000)):
            batches.append(batch)
            rows_remaining -= batch.num_rows
            if rows_remaining <= 0:
                break
        table = pa.Table.from_batches(batches).slice(0, sample_size)
        return table.to_pandas()
    return pd.read_parquet(file_path)


def download_dataset(sample_size: int | None = None, dataset_id: str = DATASET_ID) -> pd.DataFrame:
    cached_file = find_cached_dataset_file()
    if cached_file is not None:
        return read_cached_file(cached_file, sample_size)

    split_name = f"train[:{sample_size}]" if sample_size else "train"
    try:
        dataset = load_dataset(dataset_id, split=split_name) if sample_size else load_dataset(dataset_id)["train"]
    except Exception:
        dataset = (
            load_dataset(dataset_id, split=split_name, download_config=DownloadConfig(local_files_only=True))
            if sample_size
            else load_dataset(dataset_id, download_config=DownloadConfig(local_files_only=True))["train"]
        )
    return dataset.to_pandas()


def normalize_unicode_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return unicodedata.normalize("NFKC", str(value))


def clean_text_basic(value: Any) -> str:
    text = normalize_unicode_text(value).lower().strip()
    text = HTML_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" ", text)
    text = EMAIL_REGEX.sub(" ", text)
    text = PHONE_REGEX.sub(" ", text)
    text = REPEATED_PUNCT_REGEX.sub(r"\1", text)
    text = WHITESPACE_REGEX.sub(" ", text)
    return text.strip()


def clean_salary_text_basic(value: Any) -> str:
    """Làm sạch văn bản trường lương — KHÔNG áp dụng PHONE_REGEX.

    Lý do: PHONE_REGEX có thể khớp nhầm các dải số lương dạng
    'X.000.000-Y.000.000 VND' (ví dụ: '10.000.000-15.000.000') vì
    đoạn '000.000-15.000' bắt đầu bằng '0' và chứa các chữ số cách
    nhau bởi dấu chấm/gạch nối, trông giống số điện thoại.
    Hàm này chỉ dùng để phân tích cột 'salary'; các trường văn bản
    dài (job_description, requirements, benefits) vẫn dùng clean_text_basic.
    """
    text = normalize_unicode_text(value).lower().strip()
    text = HTML_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" ", text)
    text = EMAIL_REGEX.sub(" ", text)
    # Không áp dụng PHONE_REGEX để bảo toàn cấu trúc số của dải lương
    text = REPEATED_PUNCT_REGEX.sub(r"\1", text)
    text = WHITESPACE_REGEX.sub(" ", text)
    return text.strip()


def fold_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_salary_text(value: Any) -> str:
    text = fold_accents(clean_salary_text_basic(value))
    replacements = {
        "us$": "usd",
        "$/month": " usd ",
        "$ /month": " usd ",
        "$/thang": " usd ",
        "/month": " / month ",
        "/thang": " / month ",
        "tri u": "trieu",
        "trieu dong": "trieu",
        "vnd/thang": "vnd",
        "vnđ/tháng": "vnd",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("~", " - ").replace("–", " - ").replace("—", " - ")
    return WHITESPACE_REGEX.sub(" ", text).strip()


def classify_salary_pattern(text: str) -> str:
    if not text:
        return "missing"
    if any(keyword in text for keyword in AMBIGUOUS_SALARY_KEYWORDS):
        return "ambiguous"
    has_range = bool(RANGE_SPLIT_REGEX.search(text))
    has_number = bool(NUMBER_TOKEN_REGEX.search(text))
    has_usd = "usd" in text or "$" in text
    has_vnd = any(token in text for token in ["vnd", "vnđ", "trieu", "triệu", "tr", "dong", "đ"])
    if has_range and has_number:
        return "range_usd" if has_usd else "range_vnd" if has_vnd else "range_unknown"
    if has_number:
        return "single_usd" if has_usd else "single_vnd" if has_vnd else "single_unknown"
    return "invalid"


def detect_salary_mentions(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(lambda value: bool(SALARY_LEAKAGE_REGEX.search(normalize_unicode_text(value).lower())))


def parse_number_token(token: str, currency_hint: str) -> float | None:
    cleaned = token.strip()
    if not cleaned:
        return None
    if currency_hint == "usd":
        cleaned = cleaned.replace(",", "")
        if cleaned.count(".") > 1:
            cleaned = cleaned.replace(".", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", cleaned):
        return float(cleaned.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(,\d{3})+", cleaned):
        return float(cleaned.replace(",", ""))
    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        left, right = cleaned.split(",")
        if len(right) <= 2:
            cleaned = f"{left}.{right}"
    try:
        return float(cleaned.replace(",", ""))
    except ValueError:
        return None


def infer_currency(text: str) -> str | None:
    if "usd" in text or "$" in text:
        return "usd"
    if any(token in text for token in ["trieu", "triệu", "vnd", "vnđ", "dong", "đ", "tr "]):
        return "vnd"
    return None


def convert_to_million_vnd(value: float, currency: str, text: str, config: PipelineConfig) -> float:
    if currency == "usd":
        return value * config.usd_to_million_vnd
    # Dùng MILLION_REGEX (word boundary) thay vì kiểm tra chuỗi con đơn giản.
    # Lý do: " tr" in text khớp nhầm các từ tiếng Việt như 'trở lên', 'trách nhiệm',
    # 'trong' khiến các giá trị dạng '8,000,000 VND TRỞ LÊN' bị giữ nguyên
    # thay vì chia cho 1,000,000, tạo ra outlier ảo 8,000,000 triệu VNĐ.
    if MILLION_REGEX.search(text):
        return value
    if any(token in text for token in ["vnd", "vnđ", "dong", "đ"]):
        return value / 1_000_000.0
    return value


def parse_salary_row(raw_salary: Any, config: PipelineConfig) -> dict[str, Any]:
    original_text = normalize_unicode_text(raw_salary)
    normalized = normalize_salary_text(original_text)
    pattern = classify_salary_pattern(normalized)
    result = {
        "salary_raw_normalized": normalized,
        "salary_pattern": pattern,
        "salary_currency": None,
        "salary_min": np.nan,
        "salary_max": np.nan,
        "salary_expected_million_vnd": np.nan,
        "salary_parse_status": "invalid",
    }
    if pattern in {"missing", "ambiguous", "invalid"}:
        result["salary_parse_status"] = pattern
        return result

    currency = infer_currency(normalized)
    if currency is None:
        result["salary_parse_status"] = "unknown_currency"
        return result

    numbers = [parse_number_token(token, currency) for token in NUMBER_TOKEN_REGEX.findall(normalized)]
    numbers = [number for number in numbers if number is not None]
    if not numbers:
        result["salary_parse_status"] = "no_numbers"
        return result

    has_range = len(RANGE_SPLIT_REGEX.split(normalized)) > 1 and len(numbers) >= 2
    raw_min, raw_max = sorted(numbers[:2]) if has_range else (numbers[0], numbers[0])
    salary_min = convert_to_million_vnd(raw_min, currency, normalized, config)
    salary_max = convert_to_million_vnd(raw_max, currency, normalized, config)

    result.update(
        {
            "salary_currency": currency,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_expected_million_vnd": (salary_min + salary_max) / 2.0,
            "salary_parse_status": "valid",
        }
    )
    return result


def add_salary_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    parsed = df["salary"].apply(lambda value: parse_salary_row(value, config)).apply(pd.Series)
    return pd.concat([df.copy(), parsed], axis=1)


def basic_schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_ratio": [float(df[col].isna().mean()) for col in df.columns],
            "n_unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.isna().mean().sort_values(ascending=False).rename("missing_ratio").reset_index().rename(columns={"index": "column"})
    result["missing_count"] = result["column"].map(lambda col: int(df[col].isna().sum()))
    return result[["column", "missing_count", "missing_ratio"]]


def salary_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df["salary"].fillna("").astype(str).map(normalize_salary_text)
    patterns = normalized.map(classify_salary_pattern)
    result = patterns.value_counts(dropna=False).rename_axis("salary_pattern").reset_index(name="count")
    result["ratio"] = result["count"] / len(df)
    return result


def duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    near_duplicate_key = ["job_title", "company_name", "location", "salary"]
    return pd.DataFrame(
        [
            {"check": "duplicate_id", "count": int(df["id"].duplicated(keep=False).sum())},
            {"check": "duplicate_full_row", "count": int(df.duplicated(keep=False).sum())},
            {"check": "duplicate_near_key", "count": int(df.duplicated(subset=near_duplicate_key, keep=False).sum())},
        ]
    )


def text_length_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in RAW_TEXT_LENGTH_COLUMNS:
        lengths = df[column].fillna("").astype(str).str.len()
        rows.append(
            {
                "column": column,
                "mean_length": float(lengths.mean()),
                "median_length": float(lengths.median()),
                "p90_length": float(lengths.quantile(0.9)),
                "max_length": int(lengths.max()),
            }
        )
    return pd.DataFrame(rows)


def top_value_summary(df: pd.DataFrame, columns: list[str], top_n: int = 10) -> pd.DataFrame:
    frames = []
    for column in columns:
        top_values = df[column].fillna("unknown").astype(str).value_counts().head(top_n).rename_axis("value").reset_index(name="count")
        top_values.insert(0, "column", column)
        frames.append(top_values)
    return pd.concat(frames, ignore_index=True)


def salary_length_flags(series: pd.Series) -> pd.DataFrame:
    normalized = series.fillna("").astype(str).map(normalize_salary_text)
    return pd.DataFrame(
        {
            "salary_length": normalized.str.len(),
            "salary_word_count": normalized.str.split().str.len(),
            "salary_has_multiple_currencies": normalized.str.contains(r"(?:usd.*vnd|vnd.*usd|\$.*vnd|vnd.*\$)", regex=True),
            "salary_has_no_digits": ~normalized.str.contains(r"\d", regex=True),
        }
    )


def raw_rule_candidates(df: pd.DataFrame) -> dict[str, Any]:
    salary_mention_ratios = {column: float(detect_salary_mentions(df[column]).mean()) for column in LONG_TEXT_COLUMNS}
    return {
        "remove_ambiguous_salary_keywords": AMBIGUOUS_SALARY_KEYWORDS,
        "default_remove_salary_mentions_for_clustering": False,
        "high_salary_mention_columns": sorted(salary_mention_ratios, key=salary_mention_ratios.get, reverse=True),
        "top_salary_patterns": salary_pattern_summary(df).head(10).to_dict(orient="records"),
    }


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_raw_eda_figures(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    salary_pattern_summary(df).head(8).plot.bar(x="salary_pattern", y="count", ax=ax, legend=False, color="#2a6f97")
    ax.set_title("Raw salary patterns")
    save_plot(fig, output_dir / "raw_salary_patterns.png")

    for column in ["job_industry", "location", "job_type", "experience_level", "education_level", "job_position"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[column].fillna("unknown").astype(str).value_counts().head(10).sort_values().plot.barh(ax=ax, color="#5b8e7d")
        ax.set_title(f"Top 10 {column}")
        save_plot(fig, output_dir / f"top_{column}.png")

    melted = pd.DataFrame({column: df[column].fillna("").astype(str).str.len() for column in RAW_TEXT_LENGTH_COLUMNS}).melt(var_name="column", value_name="length")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=melted, x="column", y="length", ax=ax, showfliers=False, color="#a4c3b2")
    ax.set_title("Text length distribution")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, output_dir / "raw_text_length_boxplot.png")

    salary_mentions = pd.Series({column: detect_salary_mentions(df[column]).mean() for column in LONG_TEXT_COLUMNS})
    fig, ax = plt.subplots(figsize=(8, 5))
    salary_mentions.sort_values().plot.barh(ax=ax, color="#bc4749")
    ax.set_title("Salary mention ratio by text column")
    fig.tight_layout()
    fig.savefig(output_dir / "salary_mention_ratio.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "salary_leakage_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    salary_length_flags(df["salary"])["salary_length"].clip(upper=120).plot.hist(ax=ax, bins=40, color="#6d597a")
    ax.set_title("Salary string length distribution")
    save_plot(fig, output_dir / "salary_string_length_hist.png")


def run_raw_eda(config: PipelineConfig) -> pd.DataFrame:
    ensure_directories(config)
    df = download_dataset(config.sample_size, config.dataset_id)
    basic_schema_summary(df).to_csv(config.audit_dir / "raw_schema_summary.csv", index=False)
    missing_summary(df).to_csv(config.audit_dir / "raw_missing_summary.csv", index=False)
    text_length_summary(df).to_csv(config.audit_dir / "raw_text_length_summary.csv", index=False)
    salary_pattern_summary(df).to_csv(config.audit_dir / "raw_salary_pattern_summary.csv", index=False)
    duplicate_summary(df).to_csv(config.audit_dir / "raw_duplicate_summary.csv", index=False)
    top_value_summary(df, CATEGORICAL_COLUMNS).to_csv(config.audit_dir / "raw_top_values.csv", index=False)
    salary_mention_summary = pd.DataFrame(
        [{"column": column, "salary_mention_ratio": float(detect_salary_mentions(df[column]).mean())} for column in LONG_TEXT_COLUMNS]
    )
    salary_mention_summary.to_csv(config.audit_dir / "raw_salary_mention_summary.csv", index=False)
    salary_mention_summary.rename(columns={"salary_mention_ratio": "leakage_ratio"}).to_csv(
        config.audit_dir / "raw_salary_leakage_summary.csv",
        index=False,
    )
    raw_outliers = pd.concat([df[["id", "salary"]].reset_index(drop=True), salary_length_flags(df["salary"])], axis=1)
    raw_outliers.sort_values(["salary_has_multiple_currencies", "salary_length"], ascending=[False, False]).head(200).to_csv(
        config.audit_dir / "raw_salary_outlier_examples.csv",
        index=False,
    )
    with open(config.audit_dir / "raw_rule_candidates.json", "w", encoding="utf-8") as file:
        json.dump(raw_rule_candidates(df), file, ensure_ascii=False, indent=2)
    create_raw_eda_figures(df, config.figures_dir)
    return df


def mark_invalid_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_is_valid"] = (
        (df["salary_parse_status"] == "valid")
        & np.isfinite(df["salary_expected_million_vnd"])
        & (df["salary_expected_million_vnd"] > 0)
        & (df["salary_min"] <= df["salary_max"])
    )
    return df


def flag_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    valid = df.loc[df["salary_parse_status"] == "valid", "salary_expected_million_vnd"]
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    p1 = float(valid.quantile(0.01))
    p99 = float(valid.quantile(0.99))
    iqr = q3 - q1
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr
    reasons = np.where(
        df["salary_expected_million_vnd"] < max(0.0, lower_iqr),
        "below_iqr_lower",
        np.where(df["salary_expected_million_vnd"] > upper_iqr, "above_iqr_upper", ""),
    )
    percentile_reasons = np.where(
        df["salary_expected_million_vnd"] < p1,
        "below_p1",
        np.where(df["salary_expected_million_vnd"] > p99, "above_p99", ""),
    )
    flagged = df.copy()
    flagged["is_salary_outlier"] = (reasons != "") | (percentile_reasons != "")
    flagged["outlier_reason"] = np.where(
        (reasons != "") & (percentile_reasons != ""),
        reasons + "|" + percentile_reasons,
        np.where(reasons != "", reasons, percentile_reasons),
    )
    summary = {
        "q1": q1,
        "q3": q3,
        "p1": p1,
        "p99": p99,
        "iqr": iqr,
        "lower_iqr": lower_iqr,
        "upper_iqr": upper_iqr,
        "outlier_count": float(flagged["is_salary_outlier"].sum()),
    }
    return flagged, summary


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    near_duplicate_key = ["job_title", "company_name", "location", "salary"]
    flagged = df.copy()
    flagged["is_near_duplicate"] = flagged.duplicated(subset=near_duplicate_key, keep=False)
    audit = flagged.loc[flagged["is_near_duplicate"], ["id", *near_duplicate_key]].copy()
    deduped = flagged.drop_duplicates(subset=["id"]).drop_duplicates()
    return deduped, audit


def fill_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in TEXT_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .map(clean_text_basic)
            .replace({"": "unknown", "khong": "unknown", "none": "unknown", "n/a": "unknown", "null": "unknown"})
        )
    return cleaned


def remove_salary_leakage(text: Any) -> str:
    cleaned = SALARY_LEAKAGE_REGEX.sub(" ", clean_text_basic(text))
    cleaned = re.sub(r"\b(luong|thu nhap|salary)\b\s*[:\-]?", " ", cleaned)
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()
    return cleaned or "unknown"


def apply_cleaning(df: pd.DataFrame, remove_salary_mentions: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = fill_unknowns(df)
    logs = []
    for column in LONG_TEXT_COLUMNS:
        original = cleaned[column].astype(str)
        had_salary_mentions = original.map(lambda value: bool(SALARY_LEAKAGE_REGEX.search(value)))
        if remove_salary_mentions:
            cleaned[column] = original.map(remove_salary_leakage)
        logs.append(
            {
                "column": column,
                "rows_with_salary_mentions": int(had_salary_mentions.sum()),
                "salary_mention_ratio": float(had_salary_mentions.mean()),
                "salary_mentions_removed": bool(remove_salary_mentions),
            }
        )
    return cleaned, pd.DataFrame(logs)


def build_stratify_bins(series: pd.Series, n_bins: int = 10) -> pd.Series | None:
    valid = series.dropna()
    if valid.nunique() < 2:
        return None
    bins = pd.qcut(valid, q=min(n_bins, valid.nunique()), duplicates="drop")
    return bins.reindex(series.index) if bins.nunique() >= 2 else None


def split_dataset(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = build_stratify_bins(df["salary_expected_million_vnd"])
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify if stratify is not None else None,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_cleaning_figures(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="salary_expected_million_vnd", hue="split", bins=40, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_xlim(left=0)
    ax.set_title("Salary feature distribution by split")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution_by_split.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "salary_feature_distribution_by_split.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="salary_expected_million_vnd", ax=ax, color="#8d99ae", showfliers=False)
    ax.set_title("Salary feature boxplot")
    fig.tight_layout()
    fig.savefig(output_dir / "salary_target_boxplot.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "salary_feature_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    df["is_salary_outlier"].value_counts().rename(index={True: "outlier", False: "not_outlier"}).plot.pie(
        ax=ax, autopct="%1.1f%%", colors=["#d62828", "#2a9d8f"]
    )
    ax.set_ylabel("")
    ax.set_title("Outlier flag ratio")
    save_plot(fig, output_dir / "outlier_flag_ratio.png")


def parser_test_cases(config: PipelineConfig) -> pd.DataFrame:
    test_cases = [
        {"salary": "26 - 36 triệu", "expected": 31.0, "expected_status": "valid"},
        {"salary": "15 - 35 triệu", "expected": 25.0, "expected_status": "valid"},
        {"salary": "4000 usd", "expected": 100.0, "expected_status": "valid"},
        {"salary": "1200 usd", "expected": 30.0, "expected_status": "valid"},
        {"salary": "14.000.000 - 20.000.000 vnd", "expected": 17.0, "expected_status": "valid"},
        {"salary": "thỏa thuận", "expected": np.nan, "expected_status": "ambiguous"},
        {"salary": "đang cập nhật", "expected": np.nan, "expected_status": "invalid"},
    ]
    rows = []
    for case in test_cases:
        parsed = parse_salary_row(case["salary"], config)
        actual = parsed["salary_expected_million_vnd"]
        expected = case["expected"]
        if np.isnan(expected):
            value_ok = pd.isna(actual)
        else:
            value_ok = abs(float(actual) - expected) < 1e-9
        status_ok = parsed["salary_parse_status"] == case["expected_status"]
        rows.append({**case, **parsed, "value_ok": value_ok, "status_ok": status_ok, "passed": bool(value_ok and status_ok)})
    return pd.DataFrame(rows)


def salary_parse_audit(df: pd.DataFrame) -> pd.DataFrame:
    status_counts = (
        df["salary_parse_status"]
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="count")
    )
    status_counts.insert(0, "audit_type", "salary_parse_status")
    pattern_counts = (
        df["salary_pattern"]
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="count")
    )
    pattern_counts.insert(0, "audit_type", "salary_pattern")
    audit = pd.concat([status_counts, pattern_counts], ignore_index=True)
    audit["ratio"] = audit["count"] / len(df)
    return audit


def data_dictionary() -> pd.DataFrame:
    rows = [
        {"column": "salary_expected_million_vnd", "role": "numeric_feature", "description": "Expected salary in million VND, used as a clustering feature."},
        {"column": "salary_min", "role": "numeric_audit", "description": "Lower salary bound in million VND."},
        {"column": "salary_max", "role": "numeric_audit", "description": "Upper salary bound in million VND."},
        {"column": "salary_range_width", "role": "numeric_feature_candidate", "description": "Difference between salary_max and salary_min."},
        {"column": "is_salary_outlier", "role": "audit_flag", "description": "Salary outlier flag; not removed by default."},
        {"column": "job_title", "role": "text_feature", "description": "Job title text."},
        {"column": "job_description", "role": "text_feature", "description": "Job description text."},
        {"column": "requirements", "role": "text_feature", "description": "Requirements text."},
        {"column": "benefits", "role": "text_feature", "description": "Benefits text."},
        {"column": "location", "role": "categorical_feature", "description": "Job location."},
        {"column": "job_type", "role": "categorical_feature", "description": "Job type."},
        {"column": "job_industry", "role": "categorical_feature", "description": "Job industry."},
        {"column": "experience_level", "role": "categorical_feature", "description": "Experience level."},
        {"column": "education_level", "role": "categorical_feature", "description": "Education level."},
        {"column": "job_position", "role": "categorical_feature", "description": "Job position."},
    ]
    return pd.DataFrame(rows)


def processing_summary(
    raw_rows: int,
    parsed_df: pd.DataFrame,
    valid_rows_before_dedup: int,
    rows_after_dedup: int,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    invalid_counts = parsed_df["salary_parse_status"].value_counts(dropna=False).to_dict()
    rows = [
        {"metric": "raw_rows", "value": raw_rows},
        {"metric": "salary_valid_rows_before_dedup", "value": valid_rows_before_dedup},
        {"metric": "rows_after_dedup", "value": rows_after_dedup},
        {"metric": "train_rows", "value": len(raw_train)},
        {"metric": "test_rows", "value": len(raw_test)},
        {"metric": "test_size", "value": config.test_size},
        {"metric": "random_state", "value": config.random_state},
        {"metric": "remove_salary_mentions", "value": config.remove_salary_mentions},
    ]
    for status, count in invalid_counts.items():
        rows.append({"metric": f"salary_status_{status}", "value": int(count)})
    return pd.DataFrame(rows)


def train_test_salary_distribution(raw_train: pd.DataFrame, raw_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in [("train", raw_train), ("test", raw_test)]:
        salary = frame["salary_expected_million_vnd"]
        rows.append(
            {
                "split": split,
                "rows": len(frame),
                "ratio": len(frame) / (len(raw_train) + len(raw_test)),
                "salary_mean": float(salary.mean()),
                "salary_median": float(salary.median()),
                "salary_std": float(salary.std()),
                "salary_min": float(salary.min()),
                "salary_max": float(salary.max()),
            }
        )
    return pd.DataFrame(rows)


def write_dataset_outputs(raw_train: pd.DataFrame, raw_test: pd.DataFrame, clean_train: pd.DataFrame, clean_test: pd.DataFrame, config: PipelineConfig) -> None:
    raw_train.to_csv(config.raw_dir / "cluster_train_raw.csv", index=False)
    raw_test.to_csv(config.raw_dir / "cluster_test_raw.csv", index=False)
    clean_train.to_csv(config.clean_dir / "cluster_train_clean.csv", index=False)
    clean_test.to_csv(config.clean_dir / "cluster_test_clean.csv", index=False)

    # Backward-compatible names required by earlier submission instructions.
    raw_train.to_csv(config.raw_dir / "raw_data_train.csv", index=False)
    raw_test.to_csv(config.raw_dir / "raw_data_test.csv", index=False)
    clean_train.to_csv(config.clean_dir / "clean_data_train.csv", index=False)
    clean_test.to_csv(config.clean_dir / "clean_data_test.csv", index=False)


def run_processing_pipeline(config: PipelineConfig) -> dict[str, Any]:
    ensure_directories(config)
    df = download_dataset(config.sample_size, config.dataset_id)
    raw_rows = len(df)
    df = add_salary_features(df, config)
    df["salary_range_width"] = df["salary_max"] - df["salary_min"]
    df = mark_invalid_targets(df)
    valid_df = df.loc[df["target_is_valid"]].copy()
    valid_rows_before_dedup = len(valid_df)
    valid_df, outlier_summary = flag_outliers(valid_df)
    valid_df, near_duplicate_audit = deduplicate(valid_df)
    rows_after_dedup = len(valid_df)
    raw_train, raw_test = split_dataset(valid_df, config)
    clean_train, salary_mention_train = apply_cleaning(raw_train, remove_salary_mentions=config.remove_salary_mentions)
    clean_test, salary_mention_test = apply_cleaning(raw_test, remove_salary_mentions=config.remove_salary_mentions)

    salary_mention_logs = (
        pd.concat([salary_mention_train.assign(split="train"), salary_mention_test.assign(split="test")], ignore_index=True)
        .groupby("column", as_index=False)
        .agg(
            rows_with_salary_mentions=("rows_with_salary_mentions", "sum"),
            salary_mention_ratio=("salary_mention_ratio", "mean"),
            salary_mentions_removed=("salary_mentions_removed", "max"),
        )
    )

    parser_tests = parser_test_cases(config)
    if not bool(parser_tests["passed"].all()):
        failed = parser_tests.loc[~parser_tests["passed"], ["salary", "expected", "expected_status", "salary_expected_million_vnd", "salary_parse_status"]]
        raise AssertionError(f"Salary parser test cases failed:\n{failed.to_string(index=False)}")

    write_dataset_outputs(raw_train, raw_test, clean_train, clean_test, config)
    salary_mention_logs.to_csv(config.audit_dir / "clean_salary_mention_summary.csv", index=False)
    salary_mention_logs.to_csv(config.audit_dir / "clean_salary_leakage_summary.csv", index=False)
    near_duplicate_audit.to_csv(config.audit_dir / "near_duplicate_audit.csv", index=False)
    pd.DataFrame([outlier_summary]).to_csv(config.audit_dir / "salary_outlier_summary.csv", index=False)
    salary_parse_audit(df).to_csv(config.audit_dir / "salary_parse_audit.csv", index=False)
    parser_tests.to_csv(config.audit_dir / "salary_parser_tests.csv", index=False)
    processing_summary(raw_rows, df, valid_rows_before_dedup, rows_after_dedup, raw_train, raw_test, config).to_csv(
        config.audit_dir / "processing_summary.csv",
        index=False,
    )
    train_test_salary_distribution(raw_train, raw_test).to_csv(config.audit_dir / "train_test_salary_distribution.csv", index=False)
    data_dictionary().to_csv(config.audit_dir / "data_dictionary.csv", index=False)
    pd.concat([raw_train, raw_test], ignore_index=True).loc[lambda frame: frame["is_salary_outlier"]].head(200).to_csv(
        config.audit_dir / "salary_outlier_examples.csv",
        index=False,
    )
    train_test_salary_distribution(raw_train, raw_test).rename(
        columns={"salary_mean": "target_mean", "salary_median": "target_median"}
    ).to_csv(config.audit_dir / "train_test_target_distribution.csv", index=False)
    create_cleaning_figures(pd.concat([raw_train.assign(split="train"), raw_test.assign(split="test")]), config.figures_dir)
    return {
        "raw_train": raw_train,
        "raw_test": raw_test,
        "clean_train": clean_train,
        "clean_test": clean_test,
        "outlier_summary": outlier_summary,
        "salary_mention_logs": salary_mention_logs,
        "parser_tests": parser_tests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Processing pipeline for Vietnamese job clustering.")
    parser.add_argument("--mode", choices=["eda", "process"], default="process")
    parser.add_argument("--data-dir", default="artifacts")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--usd-to-vnd", type=float, default=25000.0)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--remove-salary-mentions", action="store_true", help="Optionally remove explicit salary mentions from long text columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        usd_to_vnd=args.usd_to_vnd,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_size=args.sample_size,
        remove_salary_mentions=args.remove_salary_mentions,
    )
    if args.mode == "eda":
        df = run_raw_eda(config)
        print(f"Loaded {len(df):,} rows for raw EDA.")
        print(f"Audit files saved to: {config.audit_dir}")
        print(f"Figures saved to: {config.figures_dir}")
        return

    results = run_processing_pipeline(config)
    print(f"Raw train rows: {len(results['raw_train']):,}")
    print(f"Raw test rows: {len(results['raw_test']):,}")
    print(f"Clean train rows: {len(results['clean_train']):,}")
    print(f"Clean test rows: {len(results['clean_test']):,}")
    print(f"Outlier summary: {results['outlier_summary']}")
    print(f"Parser tests passed: {bool(results['parser_tests']['passed'].all())}")


if __name__ == "__main__":
    main()

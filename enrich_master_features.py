"""Create enriched monthly and weekly Berka master tables.

The original master parquet files are preserved.  This script creates:

* ``berka_master_m_enriched.parquet``
* ``berka_master_w_enriched.parquet``

The added columns are known-calendar covariates and a contractual loan
repayment schedule.  The schedule is generated from ``loan.payments``,
``loan.date`` and ``loan.duration``; observed ``trans`` transactions are not
used to create future scheduled amounts.

Important modeling note
-----------------------
``scheduled_loan_repayment`` is a period-level contractual schedule.  When a
training row at origin ``t`` is later used to predict ``t+h``, the model
builder must still apply an as-of mask: a loan whose ``loan.date`` is after
the origin was not known at that origin.  This script creates the canonical
period schedule; it does not replace that horizon-specific leakage check.

Run from the project root:

    .\\ts\\Scripts\\python.exe enrich_master_features.py --frequency both
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MASTER_DIR = ROOT / "Berka Dataset" / "Berka Dataset" / "master_tables"
LOAN_PATH = ROOT / "Berka Dataset" / "Berka Dataset" / "loan.csv"


# Historical Czech non-working holidays for the Berka observation period.
# Dates are intentionally explicit rather than generated from a current
# holiday package, because the legal holiday set changed after this period.
HOLIDAYS_BY_YEAR: dict[int, tuple[str, ...]] = {
    1993: ("01-01", "04-12", "05-01", "05-08", "07-05", "07-06", "10-28", "12-24", "12-25", "12-26"),
    1994: ("01-01", "04-04", "05-01", "05-08", "07-05", "07-06", "10-28", "12-24", "12-25", "12-26"),
    1995: ("01-01", "04-17", "05-01", "05-08", "07-05", "07-06", "10-28", "12-24", "12-25", "12-26"),
    1996: ("01-01", "04-08", "05-01", "05-08", "07-05", "07-06", "10-28", "12-24", "12-25", "12-26"),
    1997: ("01-01", "03-31", "05-01", "05-08", "07-05", "07-06", "10-28", "12-24", "12-25", "12-26"),
    1998: ("01-01", "04-13", "05-01", "05-08", "07-05", "07-06", "10-28", "12-24", "12-25", "12-26"),
}

EXPECTED_MONTHLY_FEATURES = [
    "holiday_count",
    "has_holiday",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "scheduled_loan_repayment",
]

EXPECTED_WEEKLY_FEATURES = [
    "holiday_count",
    "has_holiday",
    "is_week_before_holiday",
    "is_week_after_holiday",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "week_sin",
    "week_cos",
    "has_month_start",
    "has_month_end",
    "scheduled_loan_repayment",
]


def exact_holiday_dates() -> pd.DatetimeIndex:
    """Return the exact 1993--1998 Czech holiday dates."""
    values = [
        pd.Timestamp(f"{year}-{month_day}")
        for year, month_days in HOLIDAYS_BY_YEAR.items()
        for month_day in month_days
    ]
    return pd.DatetimeIndex(sorted(values))


def _cyclic_pair(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode zero-based cyclic values as sine/cosine pairs."""
    phase = 2.0 * np.pi * values / period
    return np.sin(phase), np.cos(phase)


def _monthly_calendar_features(periods: pd.DatetimeIndex) -> pd.DataFrame:
    """Build monthly calendar features for unique month-start dates."""
    periods = pd.DatetimeIndex(pd.to_datetime(periods)).normalize()
    month_index = periods.month.to_numpy(dtype=float) - 1.0
    quarter_index = periods.quarter.to_numpy(dtype=float) - 1.0
    month_sin, month_cos = _cyclic_pair(month_index, 12.0)
    quarter_sin, quarter_cos = _cyclic_pair(quarter_index, 4.0)

    holiday_counts = Counter(
        holiday.to_period("M") for holiday in exact_holiday_dates()
    )
    holiday_count = np.asarray(
        [holiday_counts.get(period.to_period("M"), 0) for period in periods],
        dtype=np.int64,
    )

    return pd.DataFrame(
        {
            "period_start": periods,
            "holiday_count": holiday_count,
            "has_holiday": (holiday_count > 0).astype(np.int8),
            "month_sin": month_sin,
            "month_cos": month_cos,
            "quarter_sin": quarter_sin,
            "quarter_cos": quarter_cos,
        }
    )


def _week_start_for_date(date: pd.Timestamp) -> pd.Timestamp:
    """Return the Sunday starting the weekly period containing ``date``."""
    days_since_sunday = (date.dayofweek + 1) % 7
    return (date - pd.Timedelta(days=days_since_sunday)).normalize()


def _weekly_calendar_features(periods: pd.DatetimeIndex) -> pd.DataFrame:
    """Build Sunday-start weekly calendar features."""
    periods = pd.DatetimeIndex(pd.to_datetime(periods)).normalize()
    if not np.all(periods.dayofweek == 6):
        raise ValueError("Weekly master periods must start on Sunday")

    holiday_counts_by_week = Counter(
        _week_start_for_date(holiday) for holiday in exact_holiday_dates()
    )
    holiday_count = np.asarray(
        [holiday_counts_by_week.get(period, 0) for period in periods],
        dtype=np.int64,
    )
    holiday_count_by_period = dict(zip(periods, holiday_count))
    next_week_count = np.asarray(
        [holiday_count_by_period.get(period + pd.Timedelta(weeks=1), 0) for period in periods],
        dtype=np.int64,
    )
    previous_week_count = np.asarray(
        [holiday_count_by_period.get(period - pd.Timedelta(weeks=1), 0) for period in periods],
        dtype=np.int64,
    )

    month_index = periods.month.to_numpy(dtype=float) - 1.0
    quarter_index = periods.quarter.to_numpy(dtype=float) - 1.0
    month_sin, month_cos = _cyclic_pair(month_index, 12.0)
    quarter_sin, quarter_cos = _cyclic_pair(quarter_index, 4.0)

    # Use a date-based annual phase so 52/53-week years do not require an
    # arbitrary ISO-week convention.  The feature is named week_* because it
    # is attached to weekly rows and represents annual weekly seasonality.
    day_of_year = periods.dayofyear.to_numpy(dtype=float) - 1.0
    days_in_year = np.where(periods.is_leap_year, 366.0, 365.0)
    week_phase = day_of_year / days_in_year
    week_sin = np.sin(2.0 * np.pi * week_phase)
    week_cos = np.cos(2.0 * np.pi * week_phase)

    years = range(int(periods.min().year) - 1, int(periods.max().year) + 2)
    month_starts = {
        pd.Timestamp(year=year, month=month, day=1)
        for year in years
        for month in range(1, 13)
    }
    month_ends = {
        pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
        for year in years
        for month in range(1, 13)
    }

    has_month_start = np.zeros(len(periods), dtype=np.int8)
    has_month_end = np.zeros(len(periods), dtype=np.int8)
    for offset in range(7):
        dates = periods + pd.Timedelta(days=offset)
        has_month_start |= dates.isin(month_starts).astype(np.int8)
        has_month_end |= dates.isin(month_ends).astype(np.int8)

    return pd.DataFrame(
        {
            "period_start": periods,
            "holiday_count": holiday_count,
            "has_holiday": (holiday_count > 0).astype(np.int8),
            "is_week_before_holiday": (next_week_count > 0).astype(np.int8),
            "is_week_after_holiday": (previous_week_count > 0).astype(np.int8),
            "month_sin": month_sin,
            "month_cos": month_cos,
            "quarter_sin": quarter_sin,
            "quarter_cos": quarter_cos,
            "week_sin": week_sin,
            "week_cos": week_cos,
            "has_month_start": has_month_start,
            "has_month_end": has_month_end,
        }
    )


def _parse_berka_date(values: pd.Series) -> pd.Series:
    """Parse Berka's six-digit YYMMDD date format."""
    return pd.to_datetime(values.astype(str), format="%y%m%d")


def _first_due_period(loan_date: pd.Timestamp) -> pd.Period:
    """Return the month containing the first scheduled payment on day 12."""
    period = loan_date.to_period("M")
    if loan_date.day > 12:
        period += 1
    return period


def _loan_schedule(
    frequency: str,
    account_periods: pd.DataFrame,
) -> pd.DataFrame:
    """Create contractual loan repayment amounts by account and period.

    ``loan.payments`` is deliberately used as the planned amount.  Actual
    ``trans`` UVER amounts can be rounded, delayed, missing, or anomalous.
    """
    loans = pd.read_csv(
        LOAN_PATH,
        sep=";",
        usecols=["account_id", "date", "payments", "duration"],
    )
    loans["account_id"] = loans["account_id"].astype(int)
    loans["loan_date"] = _parse_berka_date(loans.pop("date"))
    loans["payments"] = pd.to_numeric(loans["payments"], errors="coerce")
    loans["duration"] = pd.to_numeric(loans["duration"], errors="coerce")
    loans = loans.dropna(subset=["loan_date", "payments", "duration"])
    loans["duration"] = loans["duration"].astype(int)

    records: list[dict[str, object]] = []
    for loan in loans.itertuples(index=False):
        first_period = _first_due_period(loan.loan_date)
        for offset in range(int(loan.duration)):
            due_period = first_period + offset
            due_date = due_period.to_timestamp() + pd.Timedelta(days=11)
            if frequency == "monthly":
                period_start = due_period.to_timestamp()
            else:
                period_start = _week_start_for_date(due_date)
            records.append(
                {
                    "account_id": int(loan.account_id),
                    "period_start": period_start,
                    "scheduled_loan_repayment": float(loan.payments),
                }
            )

    schedule = pd.DataFrame(records)
    if schedule.empty:
        return pd.DataFrame(
            columns=["account_id", "period_start", "scheduled_loan_repayment"]
        )

    schedule = (
        schedule.groupby(["account_id", "period_start"], as_index=False)[
            "scheduled_loan_repayment"
        ]
        .sum()
    )
    valid_keys = account_periods[["account_id", "period_start"]].drop_duplicates()
    schedule = schedule.merge(
        valid_keys,
        on=["account_id", "period_start"],
        how="inner",
        validate="many_to_one",
    )
    return schedule


def enrich_master(frequency: str, output_path: Path) -> pd.DataFrame:
    """Enrich one master table and write it to ``output_path``."""
    suffix = "m" if frequency == "monthly" else "w"
    input_path = MASTER_DIR / f"berka_master_{suffix}.parquet"
    df = pd.read_parquet(input_path)
    df["period_start"] = pd.to_datetime(df["period_start"]).dt.normalize()
    df["account_id"] = df["account_id"].astype(int)

    new_columns = (
        EXPECTED_MONTHLY_FEATURES
        if frequency == "monthly"
        else EXPECTED_WEEKLY_FEATURES
    )
    collisions = sorted(set(new_columns).intersection(df.columns))
    if collisions:
        raise ValueError(
            f"Input already contains enriched feature columns: {collisions}"
        )

    periods = pd.DatetimeIndex(sorted(df["period_start"].unique()))
    if frequency == "monthly":
        calendar = _monthly_calendar_features(periods)
    else:
        calendar = _weekly_calendar_features(periods)

    account_periods = df[["account_id", "period_start"]].drop_duplicates()
    schedule = _loan_schedule(frequency, account_periods)
    df = df.merge(calendar, on="period_start", how="left", validate="many_to_one")
    df = df.merge(
        schedule,
        on=["account_id", "period_start"],
        how="left",
        validate="one_to_one",
    )
    df["scheduled_loan_repayment"] = (
        pd.to_numeric(df["scheduled_loan_repayment"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    df = df.sort_values(["account_id", "period_start"]).reset_index(drop=True)

    if len(df) != len(account_periods.merge(df[["account_id", "period_start"]], on=["account_id", "period_start"], how="left")):
        raise AssertionError("Enrichment changed the master row count")
    if df.duplicated(["account_id", "period_start"]).any():
        raise AssertionError("Enriched master contains duplicate account-period rows")
    if not (df["has_holiday"].eq(df["holiday_count"].gt(0).astype(np.int8))).all():
        raise AssertionError("has_holiday is inconsistent with holiday_count")
    if (df["scheduled_loan_repayment"] < 0).any():
        raise AssertionError("scheduled_loan_repayment cannot be negative")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def write_metadata(output_dir: Path) -> None:
    """Write feature definitions and the exact holiday source table."""
    metadata = {
        "holiday_dates": [date.strftime("%Y-%m-%d") for date in exact_holiday_dates()],
        "monthly_features": EXPECTED_MONTHLY_FEATURES,
        "weekly_features": EXPECTED_WEEKLY_FEATURES,
        "weekly_definition": "Sunday through Saturday",
        "loan_schedule": {
            "amount_source": "loan.payments",
            "due_day": 12,
            "first_due_rule": "same month if loan date day <= 12, otherwise next month",
            "duration_source": "loan.duration",
            "actual_transactions_not_used_for_schedule": True,
            "as_of_mask_required_in_model_training": True,
        },
    }
    metadata_path = output_dir / "enriched_feature_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create enriched Berka monthly and/or weekly master tables."
    )
    parser.add_argument(
        "--frequency",
        choices=["monthly", "weekly", "both"],
        default="both",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MASTER_DIR,
        help="Directory for enriched parquet files and metadata.",
    )
    args = parser.parse_args()

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frequency in frequencies:
        suffix = "m" if frequency == "monthly" else "w"
        output_path = args.output_dir / f"berka_master_{suffix}_enriched.parquet"
        enriched = enrich_master(frequency, output_path)
        feature_columns = (
            EXPECTED_MONTHLY_FEATURES
            if frequency == "monthly"
            else EXPECTED_WEEKLY_FEATURES
        )
        unique_periods = enriched[["account_id", "period_start"]].drop_duplicates()
        print(
            f"[OK] {frequency}: rows={len(enriched):,}, "
            f"accounts={enriched['account_id'].nunique():,}, "
            f"periods={enriched['period_start'].nunique():,}, "
            f"holiday_dates_count={int(enriched[['period_start', 'holiday_count']].drop_duplicates()['holiday_count'].sum()):,}, "
            f"scheduled_loan_rows={int((enriched['scheduled_loan_repayment'] > 0).sum()):,}"
        )
        print(f"[OK] added columns: {', '.join(feature_columns)}")
    write_metadata(args.output_dir)
    print(f"[OK] metadata: {args.output_dir / 'enriched_feature_metadata.json'}")


if __name__ == "__main__":
    main()

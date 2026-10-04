"""Contractual fixed-cost schedules used by residual forecasting.

The only deterministic fixed-cost baseline enabled here is the contractual
loan repayment schedule generated from ``loan.payments``.  The ``order``
table contains standing-order amounts, but it does not contain a due date,
activation date, termination date, or explicit frequency.  Therefore an
order amount cannot be assigned safely to a future monthly/weekly period.
Likewise, ``trans.k_symbol`` describes observed transaction purpose, not a
future contractual guarantee.  Those sources are intentionally not deducted
from the target in the first residual-model experiment.

The enriched master builder already materializes the canonical loan schedule
as ``scheduled_loan_repayment``.  This module provides the same schedule
construction and the baseline helpers used by residual models, so the rule
is explicit and auditable in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Berka Dataset" / "Berka Dataset"
LOAN_PATH = DATA_DIR / "loan.csv"
ORDER_PATH = DATA_DIR / "order.csv"

SCHEDULE_COLUMN = "scheduled_loan_repayment"
FIXED_COST_SOURCES = ("loan_payments",)
EXCLUDED_FIXED_COST_SOURCES = {
    "order": "No due/start/end/frequency fields in order.csv; cannot map an amount to a future period safely.",
    "trans_k_symbol": "Observed transaction purpose is not a guaranteed future schedule.",
}


def parse_berka_date(values: pd.Series) -> pd.Series:
    """Parse Berka's six-digit YYMMDD date representation."""
    return pd.to_datetime(values.astype(str), format="%y%m%d")


def week_start_for_date(date: pd.Timestamp) -> pd.Timestamp:
    """Return the Sunday starting the weekly period containing ``date``."""
    days_since_sunday = (date.dayofweek + 1) % 7
    return (date - pd.Timedelta(days=days_since_sunday)).normalize()


def first_due_period(loan_date: pd.Timestamp) -> pd.Period:
    """Return the month containing the first due date on day 12."""
    period = loan_date.to_period("M")
    if loan_date.day > 12:
        period += 1
    return period


def load_loan_repayment_events(frequency: str) -> pd.DataFrame:
    """Expand each loan into one row per contractual due period.

    The first payment is on the 12th of the loan month when the loan starts
    on or before the 12th; otherwise it starts on the 12th of the next month.
    ``loan.duration`` determines the number of payments and
    ``loan.payments`` determines each payment amount.
    """
    if frequency not in {"monthly", "weekly"}:
        raise ValueError("frequency must be 'monthly' or 'weekly'")

    loans = pd.read_csv(
        LOAN_PATH,
        sep=";",
        usecols=["account_id", "date", "payments", "duration"],
    )
    loans["account_id"] = loans["account_id"].astype(int)
    loans["loan_date"] = parse_berka_date(loans.pop("date"))
    loans["payments"] = pd.to_numeric(loans["payments"], errors="coerce")
    loans["duration"] = pd.to_numeric(loans["duration"], errors="coerce")
    loans = loans.dropna(subset=["loan_date", "payments", "duration"])
    loans["duration"] = loans["duration"].astype(int)

    records: list[dict[str, object]] = []
    for loan in loans.itertuples(index=False):
        first_period = first_due_period(loan.loan_date)
        for offset in range(int(loan.duration)):
            due_period = first_period + offset
            due_date = due_period.to_timestamp() + pd.Timedelta(days=11)
            period_start = (
                due_period.to_timestamp()
                if frequency == "monthly"
                else week_start_for_date(due_date)
            )
            records.append(
                {
                    "account_id": int(loan.account_id),
                    "period_start": pd.Timestamp(period_start),
                    "due_date": pd.Timestamp(due_date),
                    "loan_date": pd.Timestamp(loan.loan_date),
                    SCHEDULE_COLUMN: float(loan.payments),
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "account_id",
            "period_start",
            "due_date",
            "loan_date",
            SCHEDULE_COLUMN,
        ],
    )


def build_loan_schedule(
    frequency: str,
    account_periods: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one contractual loan baseline row per account-period.

    When ``account_periods`` is supplied, only keys present in that table are
    retained.  This is useful when validating or joining to a master table.
    """
    events = load_loan_repayment_events(frequency)
    if events.empty:
        schedule = pd.DataFrame(
            columns=["account_id", "period_start", SCHEDULE_COLUMN]
        )
    else:
        schedule = (
            events.groupby(["account_id", "period_start"], as_index=False)[
                SCHEDULE_COLUMN
            ]
            .sum()
        )

    if account_periods is not None:
        keys = account_periods[["account_id", "period_start"]].drop_duplicates()
        schedule = schedule.merge(
            keys,
            on=["account_id", "period_start"],
            how="inner",
            validate="many_to_one",
        )
    return schedule


def fixed_cost_policy() -> dict[str, object]:
    """Return the fixed-cost inclusion/exclusion policy for metadata."""
    return {
        "included_sources": list(FIXED_COST_SOURCES),
        "included_feature": SCHEDULE_COLUMN,
        "excluded_sources": EXCLUDED_FIXED_COST_SOURCES.copy(),
        "loan_amount_source": "loan.payments",
        "loan_due_day": 12,
        "loan_first_due_rule": "same month if loan date day <= 12, otherwise next month",
        "loan_duration_source": "loan.duration",
    }


def fixed_cost_baseline(
    known_features: pd.DataFrame,
    flow_name: str,
) -> pd.Series:
    """Return the deterministic baseline for one flow.

    Loans are an outflow baseline.  No deterministic inflow schedule is
    currently enabled, so inflow's baseline is zero.
    """
    if flow_name == "inflow":
        return pd.Series(0.0, index=known_features.index, dtype=float)
    if flow_name == "outflow":
        if SCHEDULE_COLUMN not in known_features.columns:
            raise ValueError(f"Known features must contain {SCHEDULE_COLUMN}")
        return pd.to_numeric(known_features[SCHEDULE_COLUMN], errors="coerce").fillna(0.0)
    raise ValueError(f"Unsupported flow: {flow_name}")


def fixed_cost_net_baseline(known_features: pd.DataFrame) -> pd.Series:
    """Return fixed-cost contribution to net flow: inflow minus outflow."""
    return -fixed_cost_baseline(known_features, "outflow")


def summarize_order_commitments() -> pd.DataFrame:
    """Load standing-order amounts for diagnostics, not forecasting baseline.

    This helper deliberately does not expand orders into future periods.  It
    makes the limitation explicit while allowing a later order-schedule
    experiment if reliable timing metadata is added.
    """
    orders = pd.read_csv(ORDER_PATH, sep=";")
    orders["account_id"] = orders["account_id"].astype(int)
    orders["amount"] = pd.to_numeric(orders["amount"], errors="coerce")
    orders["k_symbol"] = orders["k_symbol"].fillna("unspecified")
    return orders


if __name__ == "__main__":
    for frequency in ("monthly", "weekly"):
        schedule = build_loan_schedule(frequency)
        print(
            f"{frequency}: events={len(schedule):,}, "
            f"accounts={schedule['account_id'].nunique() if not schedule.empty else 0:,}, "
            f"total_scheduled={schedule[SCHEDULE_COLUMN].sum() if not schedule.empty else 0:,.2f}"
        )
    print(fixed_cost_policy())

"""The canonical data contract. Every source (5G testbed, synthetic stand-in, BGP) is converted
to these tables, and everything downstream reads only these tables.

raw.parquet   long format: run_id, ts (Unix ms, UTC), source, entity, metric, value
labels.csv    one row per injected fault, times are when the impairment was ACTIVE
"""
from __future__ import annotations

import pandas as pd

RAW_COLUMNS = {
    "run_id": "string",
    "ts": "int64",
    "source": "string",
    "entity": "string",
    "metric": "string",
    "value": "float64",
}

LABEL_COLUMNS = [
    "fault_id", "type", "severity", "params", "target", "injector",
    "t_applied", "t_removed", "status",
]

FAULT_TYPES = {"delay", "loss", "link_down", "cpu", "mem", "pod_kill", "flood", "bgp_event"}
SEVERITIES = {"mild", "strong", "event"}
STATUSES = {"ok", "failed"}


class SchemaError(ValueError):
    pass


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(f"raw table missing columns {sorted(missing)}")
    df = df[list(RAW_COLUMNS)].astype(RAW_COLUMNS)
    if df["ts"].lt(946684800000).any():  # 2000-01-01: catches seconds passed as milliseconds
        raise SchemaError("ts must be Unix milliseconds (found values before 2000-01-01)")
    dup = df.duplicated(["run_id", "ts", "entity", "metric"])
    if dup.any():
        raise SchemaError(f"{int(dup.sum())} duplicate (run_id, ts, entity, metric) rows")
    return df


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(f"labels missing columns {sorted(missing)}")
    if len(df) == 0:
        return df[LABEL_COLUMNS]
    bad_type = set(df["type"]) - FAULT_TYPES
    if bad_type:
        raise SchemaError(f"unknown fault types {bad_type}")
    bad_sev = set(df["severity"]) - SEVERITIES
    if bad_sev:
        raise SchemaError(f"unknown severities {bad_sev}")
    bad_status = set(df["status"]) - STATUSES
    if bad_status:
        raise SchemaError(f"unknown statuses {bad_status}")
    ok = df[df["status"] == "ok"]
    if (ok["t_removed"] < ok["t_applied"]).any():
        raise SchemaError("t_removed before t_applied")
    if ok["fault_id"].duplicated().any():
        raise SchemaError("duplicate fault_id")
    return df[LABEL_COLUMNS]


def column_name(entity: str, metric: str) -> str:
    """Wide-format column name used after pivoting."""
    return f"{entity}.{metric}"

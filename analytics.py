from __future__ import annotations

import pandas as pd


def build_daily_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_barking_sec": 0.0,
            "total_barking_min": 0.0,
            "event_count": 0,
            "avg_duration": 0.0,
            "longest_event": 0.0,
            "hourly_hist": [0] * 24,
            "most_common_hour": None,
            "density_curve": [0.0] * 24,
        }

    hourly = df.groupby("hour")["duration_sec"].sum().reindex(range(24), fill_value=0)
    count_by_hour = df.groupby("hour").size().reindex(range(24), fill_value=0)
    total = float(df["duration_sec"].sum())
    return {
        "total_barking_sec": total,
        "total_barking_min": total / 60.0,
        "event_count": int(len(df)),
        "avg_duration": float(df["duration_sec"].mean()),
        "longest_event": float(df["duration_sec"].max()),
        "hourly_hist": hourly.astype(float).tolist(),
        "most_common_hour": int(count_by_hour.idxmax()) if len(df) else None,
        "density_curve": (hourly / max(total, 1e-6)).tolist(),
    }


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "duration_sec"])
    out = df.groupby("day", as_index=False)["duration_sec"].sum().sort_values("day")
    return out

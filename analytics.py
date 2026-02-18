from __future__ import annotations

from datetime import datetime, timedelta

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


def build_thunder_summary(df: pd.DataFrame) -> dict:
    """Build daily summary statistics for thunder events.
    
    Args:
        df: DataFrame with thunder event data
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {
            "total_thunder_sec": 0.0,
            "total_thunder_min": 0.0,
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
        "total_thunder_sec": total,
        "total_thunder_min": total / 60.0,
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


def analyze_bark_thunder_correlation(bark_df: pd.DataFrame, thunder_df: pd.DataFrame, window_minutes: int = 10) -> dict:
    """Analyze temporal correlation between bark and thunder events.
    
    Args:
        bark_df: DataFrame with bark events (must have start_ts, end_ts columns)
        thunder_df: DataFrame with thunder events (must have start_ts, end_ts columns)
        window_minutes: Time window in minutes to consider events as correlated
        
    Returns:
        Dictionary with correlation statistics including:
        - overlapping_events: Number of bark events during/near thunder
        - bark_during_thunder_ratio: Ratio of barks during thunder periods
        - avg_barks_per_thunder: Average bark events per thunder event
        - temporal_matches: List of matched event pairs with timestamps
    """
    if bark_df.empty or thunder_df.empty:
        return {
            "overlapping_events": 0,
            "bark_during_thunder_ratio": 0.0,
            "avg_barks_per_thunder": 0.0,
            "temporal_matches": [],
            "bark_frequency_change": 0.0,
        }
    
    bark_df = bark_df.copy()
    thunder_df = thunder_df.copy()
    
    bark_df["start_dt"] = pd.to_datetime(bark_df["start_ts"])
    bark_df["end_dt"] = pd.to_datetime(bark_df["end_ts"])
    thunder_df["start_dt"] = pd.to_datetime(thunder_df["start_ts"])
    thunder_df["end_dt"] = pd.to_datetime(thunder_df["end_ts"])
    
    window_delta = timedelta(minutes=window_minutes)
    matches = []
    barks_near_thunder = 0
    
    for _, thunder_event in thunder_df.iterrows():
        thunder_start = thunder_event["start_dt"]
        thunder_end = thunder_event["end_dt"]
        window_start = thunder_start - window_delta
        window_end = thunder_end + window_delta
        
        overlapping_barks = bark_df[
            (bark_df["start_dt"] <= window_end) & (bark_df["end_dt"] >= window_start)
        ]
        
        for _, bark_event in overlapping_barks.iterrows():
            matches.append({
                "bark_start": bark_event["start_ts"],
                "bark_end": bark_event["end_ts"],
                "thunder_start": thunder_event["start_ts"],
                "thunder_end": thunder_event["end_ts"],
            })
            barks_near_thunder += 1
    
    bark_during_thunder_ratio = barks_near_thunder / len(bark_df) if len(bark_df) > 0 else 0.0
    avg_barks_per_thunder = barks_near_thunder / len(thunder_df) if len(thunder_df) > 0 else 0.0
    
    bark_frequency_change = ((1.0 - bark_during_thunder_ratio) * 100) if bark_during_thunder_ratio < 1.0 else 0.0
    
    return {
        "overlapping_events": barks_near_thunder,
        "bark_during_thunder_ratio": bark_during_thunder_ratio,
        "avg_barks_per_thunder": avg_barks_per_thunder,
        "temporal_matches": matches,
        "bark_frequency_change": bark_frequency_change,
    }

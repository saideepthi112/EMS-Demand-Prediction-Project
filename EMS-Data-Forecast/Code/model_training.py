#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMS forecasting over multiple formatted CSV batches.

- Auto-detects or synthesizes a timestamp (supports date+hour or year/month/day/hour).
- Uses spatial (ZIP index), temporal (hour/day cyclical), and incident numeric features.
- Trains LSTM, TCN, and Transformer models.
- Robust against NaNs/Infs, zero-variance scaling, and massive window counts.
- Skips files that fail and continues with the rest.
- Writes:
    results/per_file_metrics.csv
    results/summary_by_model.csv

Recommended first run on a login node:
    python -u ems_train.py \
      --data_glob "formatted_batch_*.csv" \
      --limit_files 10 \
      --seq_len 12 --horizon 1 \
      --batch_size 64 --epochs 3 \
      --models lstm tcn transformer \
      --results_dir results --logs_dir logs
"""

import os, glob, argparse, math, warnings, json, sys
from typing import List, Tuple, Optional
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------- Column candidates -----------------
TS_CANDIDATES  = [
    "timestamp","datetime","date_time","datehour","time","pickup_time","pickup_hour",
    "created_at","occured_on_datetime","call_datetime","hour_ts","event_time"
]
ZIP_CANDIDATES = [
    "zip","zipcode","zip_code","postal_code","zip5","incident_zip","origin_zip","destination_zip"
]
Y_CANDIDATES   = ["y","ems_calls","total_calls","call_count","incident_count","calls","num_calls"]

np.random.seed(1337); torch.manual_seed(1337)

# ----------------- Helpers -----------------
def guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in low: return low[key]
    return None

def parse_timestamp_col(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    try: ts = ts.tz_convert(None)
    except Exception: ts = ts.dt.tz_localize(None)
    return ts

def synthesize_timestamp_if_needed(df: pd.DataFrame) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    # date + hour
    date_keys = ["date","service_date","event_date","call_date"]
    hour_keys = ["hour","hr","hour_of_day","hod"]
    date_col = next((low.get(k) for k in date_keys if k in low), None)
    hour_col = next((low.get(k) for k in hour_keys if k in low), None)
    if date_col and hour_col:
        tsd = pd.to_datetime(df[date_col], errors="coerce")
        hrs = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
        df["__ts"] = tsd + pd.to_timedelta(hrs, unit="h")
        return "__ts"
    # year/month/day/hour
    y_keys = ["year","yr","yyyy"]; m_keys = ["month","mm"]; d_keys = ["day","dd","dom","day_of_month"]
    ycol = next((low.get(k) for k in y_keys if k in low), None)
    mcol = next((low.get(k) for k in m_keys if k in low), None)
    dcol = next((low.get(k) for k in d_keys if k in low), None)
    hcol = hour_col or next((low.get(k) for k in hour_keys if k in low), None)
    if ycol and mcol and dcol and hcol:
        base = pd.to_datetime(dict(year=df[ycol], month=df[mcol], day=df[dcol]), errors="coerce")
        hrs  = pd.to_numeric(df[hcol], errors="coerce").fillna(0).astype(int)
        df["__ts"] = base + pd.to_timedelta(hrs, unit="h")
        return "__ts"
    # stringy combined
    dh_keys = ["datehour","datetime_str","timestamp_str"]
    dhcol = next((low.get(k) for k in dh_keys if k in low), None)
    if dhcol:
        df["__ts"] = pd.to_datetime(df[dhcol], errors="coerce")
        return "__ts"
    return None

def cyclical_time_features(ts: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(ts)
    hour = ts.dt.hour.values
    dow  = ts.dt.dayofweek.values
    return pd.DataFrame({
        "hour_sin": np.sin(2*np.pi*hour/24),
        "hour_cos": np.cos(2*np.pi*hour/24),
        "dow_sin":  np.sin(2*np.pi*dow/7),
        "dow_cos":  np.cos(2*np.pi*dow/7),
    }, index=ts.index)

def infer_target(df: pd.DataFrame, y_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if y_col and y_col in df.columns: return df, y_col
    for c in Y_CANDIDATES:
        if c in df.columns: return df, c
    # fallback: sum numeric incident-type columns (exclude obvious non-incident)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"lat","lon","latitude","longitude","x","y","_zip_idx","is_holiday","population","pop"}
    ysrc = [c for c in numeric if c not in exclude]
    if not ysrc:
        raise ValueError("Could not infer target y. Pass --target_col or ensure numeric incident counts exist.")
    out = df.copy(); out["y"] = out[ysrc].sum(axis=1)
    return out, "y"

def clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

# ----------------- Dataset & windowing -----------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def build_sequences(df, seq_len, horizon, ts_col, zip_col, y_col, feature_cols,
                    scaler=None, fit_scaler=True, max_windows: int = 500_000, stride: int = 1):
    """Build sliding windows per ZIP with label-aligned indexing and OOM safeguards."""
    # --- scale features safely, preserving index alignment ---
    class _SafeStd:
        def __init__(self): self.mean_ = None; self.std_ = None
        def fit(self, X):
            self.mean_ = np.nanmean(X, axis=0)
            self.std_  = np.nanstd(X, axis=0)
            self.std_[self.std_ < 1e-12] = 1.0
            return self
        def transform(self, X):
            Z = (X - self.mean_) / self.std_
            return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = scaler or _SafeStd()
    Xdf = df[feature_cols].copy()
    Xvals = scaler.fit(Xdf.values).transform(Xdf.values) if fit_scaler else scaler.transform(Xdf.values)
    Xdf[feature_cols] = Xvals
    Xdf[feature_cols] = np.nan_to_num(Xdf[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

    # IMPORTANT: keep y as a Series so we can .loc with label indices
    y_series = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).clip(lower=0)

    # --- decide stride to cap total windows (avoid OOM) ---
    total = 0
    for _, g in df.groupby(zip_col, sort=False):
        total += max(0, len(g) - seq_len - horizon + 1)
    if total > max_windows:
        stride = max(stride, math.ceil(total / max_windows))
        print(f"[info] windows={total:,} > {max_windows:,} → using stride={stride}", flush=True)

    # --- build windows per ZIP (label-aligned) ---
    X_list, y_list = [], []
    for _, g in df.groupby(zip_col, sort=False):
        g = g.sort_values(ts_col)
        idx = g.index  # label index
        Xg = Xdf.loc[idx, feature_cols].to_numpy()
        yg = y_series.loc[idx].to_numpy()

        end = len(g) - seq_len - horizon + 1
        if end <= 0:
            continue
        for i in range(0, end, stride):
            X_list.append(Xg[i:i+seq_len])
            y_list.append(yg[i+seq_len+horizon-1])

    if not X_list:
        raise ValueError("No sequences built (seq_len/horizon too large for this file).")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, scaler

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

# ----------------- Models -----------------
class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, dropout=0.1):
        super().__init__()
        p = (k-1)*d
        self.net = nn.Sequential(
            nn.ConstantPad1d((p,0),0), nn.Conv1d(in_ch, out_ch, k, dilation=d), nn.ReLU(), nn.Dropout(dropout),
            nn.ConstantPad1d((p,0),0), nn.Conv1d(out_ch, out_ch, k, dilation=d), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x):
        y = self.net(x); r = self.down(x); return self.act(y + r)

class TCNReg(nn.Module):
    def __init__(self, in_dim, channels=[64,64,64], k=3, dropout=0.1):
        super().__init__()
        layers = []; c = in_dim
        for i, c_out in enumerate(channels):
            layers.append(TemporalBlock(c, c_out, k=k, d=2**i, dropout=dropout))
            c = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, 1))
    def forward(self, x):
        x = x.transpose(1,2)  # (B,T,F)->(B,F,T)
        z = self.tcn(x)
        return self.head(z).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerReg(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=8, layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))
    def forward(self, x):
        x = self.proj(x); x = self.pos(x); x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)

# ----------------- Train / Eval -----------------
def train_one(model, train_loader, val_loader, device, epochs=3, lr=1e-3, patience=3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    best_val = float("inf"); best = None; bad = 0
    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(Xb); loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()*len(Xb)
        tr_loss /= max(1, len(train_loader.dataset))
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                va_loss += loss_fn(model(Xb), yb).item()*len(Xb)
        va_loss /= max(1, len(val_loader.dataset))
        print(f"Epoch {ep:02d}/{epochs}  train_MAE={tr_loss:.4f}  val_MAE={va_loss:.4f}", flush=True)
        if not (np.isfinite(tr_loss) and np.isfinite(va_loss)):
            print("[warn] non-finite loss detected; continuing.", flush=True)
        if va_loss + 1e-6 < best_val:
            best_val = va_loss; bad = 0
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: print("Early stop.", flush=True); break
    if best is not None: model.load_state_dict(best)
    return model

@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval().to(device); y_all, p_all = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device); p_all.append(model(Xb).cpu().numpy()); y_all.append(yb.numpy())
    y = np.concatenate(y_all); p = np.concatenate(p_all)
    return {"MSE": float(mean_squared_error(y,p)),
            "MAE": float(mean_absolute_error(y,p)),
            "R2":  float(r2_score(y,p)) if len(np.unique(y))>1 else float("nan")}

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", default="formatted_batch_*.csv")
    ap.add_argument("--limit_files", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--models", nargs="+", default=["lstm","tcn","transformer"])
    ap.add_argument("--timestamp_col", default=None)
    ap.add_argument("--zip_col", default=None)
    ap.add_argument("--target_col", default=None)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True); os.makedirs(args.logs_dir, exist_ok=True)

    # Map "tcm" -> "tcn" if user typo
    args.models = [("tcn" if m.lower()=="tcm" else m.lower()) for m in args.models]

    # polite threading on login nodes
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    files = sorted(glob.glob(args.data_glob))
    if args.limit_files and len(files) > args.limit_files: files = files[:args.limit_files]
    if not files: print(f"No files match {args.data_glob}", file=sys.stderr); sys.exit(2)

    per_file_rows = []; failed = []

    for f in files:
        print(f"\n=== Processing {os.path.basename(f)} ===", flush=True)
        try:
            df = pd.read_csv(f)

            # timestamp
            ts_col = args.timestamp_col or guess_col(df, TS_CANDIDATES)
            if ts_col is None: ts_col = synthesize_timestamp_if_needed(df)
            if ts_col is None: raise ValueError("No timestamp column and could not synthesize (need date+hour or y/m/d/hour).")
            df[ts_col] = parse_timestamp_col(df[ts_col]); df = df.dropna(subset=[ts_col])

            # zip
            zip_col = args.zip_col or guess_col(df, ZIP_CANDIDATES)
            if zip_col is None:
                zcands = [c for c in df.columns if "zip" in c.lower()]
                if zcands: zip_col = zcands[0]
                else: raise ValueError("No ZIP-like column found.")

            # target
            df, y_col = infer_target(df, args.target_col)

            # temporal features
            tf = cyclical_time_features(df[ts_col]);  [df.__setitem__(c, tf[c].values) for c in tf.columns]

            # numeric clean + spatial code
            df["_zip_idx"] = df[zip_col].astype("category").cat.codes
            numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
            if y_col in numeric_all:
                numeric_all = [c for c in numeric_all if c != y_col] + [y_col]
            df = clean_numeric(df, numeric_all)
            df[y_col] = df[y_col].clip(lower=0)

            # features
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != y_col]

            # time split
            df = df.sort_values([ts_col, zip_col]).reset_index(drop=True)
            n = len(df); tss = df[ts_col]
            if n < 100:
                test_cut = tss.iloc[int(0.8*n)]; val_cut = tss.iloc[int(0.7*n)]
            else:
                test_cut = tss.iloc[int((1.0-args.test_ratio)*n)]
                val_cut  = tss.iloc[int((1.0-args.test_ratio-args.val_ratio)*n)]
            train_df = df[df[ts_col] <= val_cut].copy()
            val_df   = df[(df[ts_col] > val_cut) & (df[ts_col] <= test_cut)].copy()
            test_df  = df[df[ts_col] > test_cut].copy()

            # sequences (auto-stride if huge)
            Xtr, ytr, scaler = build_sequences(train_df, args.seq_len, args.horizon, ts_col, zip_col, y_col, feature_cols, None, True)
            Xva, yva, _      = build_sequences(val_df,   args.seq_len, args.horizon, ts_col, zip_col, y_col, feature_cols, scaler, False)
            Xte, yte, _      = build_sequences(test_df,  args.seq_len, args.horizon, ts_col, zip_col, y_col, feature_cols, scaler, False)

            # sanity: no NaNs/Infs
            for arr, name in [(Xtr,"Xtr"),(ytr,"ytr"),(Xva,"Xva"),(yva,"yva"),(Xte,"Xte"),(yte,"yte")]:
                if not np.isfinite(arr).all():
                    raise ValueError(f"Non-finite values in {name} after cleaning.")

            train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(SeqDataset(Xva, yva), batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(SeqDataset(Xte, yte), batch_size=args.batch_size, shuffle=False)

            in_dim = Xtr.shape[-1]
            for m in args.models:
                mm = m.lower()
                if mm == "lstm": model = LSTMReg(in_dim)
                elif mm in ("tcn","tcm"): model = TCNReg(in_dim)
                elif mm == "transformer": model = TransformerReg(in_dim)
                else: print(f"Unknown model {m} skipped."); continue

                print(f"Training {mm.upper()} on {os.path.basename(f)} ...", flush=True)
                model = train_one(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=3)
                metrics = eval_metrics(model, test_loader, device)
                row = {"file": os.path.basename(f), "model": mm.upper(), **metrics}
                with open(os.path.join(args.logs_dir, f"metrics_{os.path.basename(f)}_{mm}.json"), "w") as fh:
                    json.dump(row, fh, indent=2)
                print(f"→ {row}", flush=True)
                per_file_rows.append(row)

        except Exception as e:
            failed.append(f"{os.path.basename(f)} FAILED: {e}")
            print(f"{os.path.basename(f)} FAILED: {e}", file=sys.stderr, flush=True)
            continue

    if not per_file_rows:
        print("No successful runs; exiting.", file=sys.stderr)
        if failed: print("\n".join(failed), file=sys.stderr)
        sys.exit(3)

    per_file = pd.DataFrame(per_file_rows)
    per_file.to_csv(os.path.join(args.results_dir, "per_file_metrics.csv"), index=False)

    summary = (per_file.groupby("model")
               .agg(MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
                    MSE_mean=("MSE","mean"), MSE_std=("MSE","std"),
                    R2_mean =("R2","mean"),  R2_std =("R2","std"))
               .reset_index())
    summary.to_csv(os.path.join(args.results_dir, "summary_by_model.csv"), index=False)

    print("\n=== Summary by model ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    if failed:
        print("\nSkipped files:", flush=True)
        for m in failed: print(" - " + m, flush=True)

if __name__ == "__main__":
    main()

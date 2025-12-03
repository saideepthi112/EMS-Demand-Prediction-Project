# full_script_save_models.py
"""
EMS hourly Call_Count forecasting (regression) with 4 models:
 - MLP (flattened)
 - TCN (Temporal Convolutional Network)
 - LSTM
 - Transformer

This version saves each trained model along with scaler and metadata.
"""

import os
import json
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

# ------------------------
# REPRODUCIBILITY & DEVICE
# ------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# HYPERPARAMETERS
# ------------------------
CSV_PATH = "preprocessed data sample.csv"    # <- set path to your CSV
SEQ_LEN = 24
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
MODEL_OUTPUT_DIR = "models"     # output dir to save models & artifacts

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ------------------------
# UTIL: infer target, preprocess
# ------------------------
def infer_target_column(df: pd.DataFrame) -> str:
    if 'Call_Count' in df.columns:
        return 'Call_Count'
    exclude = {'ZIP_CODE', 'Date', 'Hour'}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found to infer target. Please provide 'Call_Count'.")
    return numeric_cols[-1]

def load_and_preprocess(csv_path: str) -> Tuple[pd.DataFrame, List[str], str]:
    df = pd.read_csv(csv_path)
    if 'Date' not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Hour' not in df.columns:
        raise ValueError("CSV must contain an 'Hour' column (0-23).")
    df = df.sort_values(['ZIP_CODE', 'Date', 'Hour']).reset_index(drop=True)
    df = df.fillna(0)
    # cyclical hour features
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

    target_col = infer_target_column(df)
    print(f"Using target column: {target_col}")

    # One-hot encode BOROUGH and low-cardinality object columns
    cat_cols = []
    if 'BOROUGH' in df.columns:
        cat_cols.append('BOROUGH')
    for c in df.select_dtypes(include=['object']).columns:
        if c not in cat_cols and c not in ['ZIP_CODE']:
            if df[c].nunique() <= 50:
                cat_cols.append(c)
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    meta_cols = {'ZIP_CODE', 'Date', 'Hour', target_col}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    return df, feature_cols, target_col

# ------------------------
# Time-based train/val/test splits by Date
# ------------------------
def build_time_splits(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    all_dates = sorted(df['Date'].dt.date.unique())
    n = len(all_dates)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_dates = set(all_dates[:train_end])
    val_dates = set(all_dates[train_end:val_end])
    test_dates = set(all_dates[val_end:])
    return {'train': train_dates, 'val': val_dates, 'test': test_dates}

# ------------------------
# Dataset (sliding windows grouped by ZIP_CODE)
# ------------------------
class EMSTimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len=24, allowed_target_dates: set = None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col

        X_windows = []
        y_targets = []
        metas = []

        for zip_code, group in df.groupby('ZIP_CODE'):
            g = group.sort_values(['Date', 'Hour']).reset_index(drop=True)
            n = len(g)
            if n <= seq_len:
                continue
            for i in range(0, n - seq_len):
                target_row = g.loc[i + seq_len]
                target_date = target_row['Date'].date()
                if allowed_target_dates is not None and target_date not in allowed_target_dates:
                    continue
                x_win = g.loc[i:i+seq_len-1, feature_cols].values.astype('float32')
                y_val = float(target_row[self.target_col])
                X_windows.append(x_win)
                y_targets.append(y_val)
                metas.append((zip_code, str(target_date), int(target_row['Hour'])))
        if len(X_windows) == 0:
            self.X = np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.float32)
        else:
            self.X = np.stack(X_windows)
            self.y = np.array(y_targets, dtype=np.float32)
        self.meta = metas

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ------------------------
# Models
# ------------------------
class MLPModel(nn.Module):
    def __init__(self, seq_len, n_features, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
        return self.net(x).squeeze(-1)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[..., :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, n_features, num_channels=[64,64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, 1)
    def forward(self, x):
        x = x.transpose(1,2)
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)

# ------------------------
# Training utilities
# ------------------------
def train_one_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        opt.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)
            total_mae += torch.abs(preds - y).sum().item()
    mse = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else float('nan')
    mae = total_mae / len(loader.dataset) if len(loader.dataset) > 0 else float('nan')
    return mse, mae

def fit_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    for ep in range(1, epochs+1):
        train_mse = train_one_epoch(model, train_loader, opt, criterion)
        val_mse, val_mae = evaluate(model, val_loader, criterion)
        print(f"Epoch {ep}/{epochs} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # load best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state

# ------------------------
# Save utilities
# ------------------------
def save_model_artifacts(model_name: str,
                         model_obj: nn.Module,
                         best_state_dict: dict,
                         scaler: StandardScaler,
                         feature_cols: List[str],
                         target_col: str,
                         seq_len: int,
                         output_dir: str = MODEL_OUTPUT_DIR,
                         extra_config: dict = None):
    os.makedirs(output_dir, exist_ok=True)
    base_name = model_name.replace(" ", "_")
    # Save best checkpoint (state_dict + metadata)
    checkpoint = {
        'model_name': model_name,
        'state_dict': best_state_dict if best_state_dict is not None else model_obj.state_dict(),
        'feature_cols': feature_cols,
        'target_col': target_col,
        'seq_len': seq_len,
        'n_features': len(feature_cols),
        'extra_config': extra_config or {}
    }
    best_path = os.path.join(output_dir, f"{base_name}_best.pt")
    torch.save(checkpoint, best_path)

    # Save final model state_dict as backup
    final_path = os.path.join(output_dir, f"{base_name}_final.pt")
    torch.save({'model_name': model_name, 'state_dict': model_obj.state_dict()}, final_path)

    # Save scaler with joblib
    scaler_path = os.path.join(output_dir, f"{base_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print(f"Saved checkpoint: {best_path}")
    print(f"Saved final state: {final_path}")
    print(f"Saved scaler: {scaler_path}")

    return {'best': best_path, 'final': final_path, 'scaler': scaler_path}

# ------------------------
# MAIN pipeline
# ------------------------
def main():
    print("Loading and preprocessing...")
    df, feature_cols, target_col = load_and_preprocess(CSV_PATH)
    print(f"Rows: {len(df)}, features: {len(feature_cols)}, target: {target_col}")

    splits = build_time_splits(df, train_ratio=0.7, val_ratio=0.15)
    print(f"Date splits -> train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    # Fit scaler on training rows only to avoid data leakage
    train_mask = df['Date'].dt.date.isin(splits['train'])
    if train_mask.sum() == 0:
        raise RuntimeError("No training rows found; check Date ranges or CSV contents.")
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols].values)
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols].values)

    # Create datasets
    train_ds = EMSTimeSeriesDataset(df_scaled, feature_cols, target_col, seq_len=SEQ_LEN, allowed_target_dates=splits['train'])
    val_ds   = EMSTimeSeriesDataset(df_scaled, feature_cols, target_col, seq_len=SEQ_LEN, allowed_target_dates=splits['val'])
    test_ds  = EMSTimeSeriesDataset(df_scaled, feature_cols, target_col, seq_len=SEQ_LEN, allowed_target_dates=splits['test'])

    print(f"Sequences -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("No training sequences generated. Check SEQ_LEN and continuity per ZIP_CODE.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_features = len(feature_cols)
    print("n_features:", n_features)

    # Save metadata JSON (feature cols + target)
    metadata = {'feature_cols': feature_cols, 'target_col': target_col, 'seq_len': SEQ_LEN}
    metadata_path = os.path.join(MODEL_OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")

    # Models to train
    models = {
        'MLP': MLPModel(SEQ_LEN, n_features, hidden=256),
        'TCN': TCNModel(n_features, num_channels=[64,64], kernel_size=3, dropout=0.2),
        'LSTM': LSTMModel(n_features, hidden_size=128, num_layers=2, dropout=0.2),
        'Transformer': TransformerModel(n_features, d_model=128, nhead=4, num_layers=2)
    }

    saved_paths = {}
    trained_models = {}

    for name, model in models.items():
        print("\n" + "="*60)
        print(f"Training {name}...")
        try:
            trained, best_state = fit_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
            trained_models[name] = trained
            # Evaluate on test
            test_mse, test_mae = evaluate(trained, test_loader, nn.MSELoss())
            print(f"Test {name} -> MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

            # Save model artifacts
            extra_cfg = {
                'model_type': name,
                'model_kwargs': {
                    'seq_len': SEQ_LEN,
                    'n_features': n_features
                }
            }
            paths = save_model_artifacts(name, trained, best_state, scaler, feature_cols, target_col, SEQ_LEN, extra_config=extra_cfg)
            saved_paths[name] = paths

        except Exception as e:
            print(f"Training failed for {name}: {e}")

    # Optionally save a tiny summary JSON of saved files
    summary_path = os.path.join(MODEL_OUTPUT_DIR, "saved_models_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(saved_paths, f, indent=2)
    print(f"\nSaved models summary at: {summary_path}")
    print("All done.")

if __name__ == "__main__":
    main()

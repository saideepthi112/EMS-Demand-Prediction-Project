"""
train_ts_models.py

Trains 4 time-series models (MLP, TCN, LSTM, Transformer) using past 24 hours to predict next hour call_count.

Usage:
    python train_ts_models.py --data_path path/to/your.csv --target_col "call count"

Adjust hyperparams in the main() function as needed.
"""

import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Utilities & Dataset
# -----------------------

def parse_and_preprocess(df: pd.DataFrame,
                         target_col: str,
                         datetime_cols: Tuple[str, str] = ("Date", "Hour")) -> pd.DataFrame:
    """
    - Ensure rows are sorted by ZIP_CODE, Date, Hour
    - Convert Date+Hour to a proper datetime if needed
    - Convert boolean-like columns to numeric
    - One-hot encode categorical columns (BOROUGH) and ensure target exists
    """
    df = df.copy()

    # If Hour is integer 0-23 and Date is YYYY-MM-DD string, combine to timestamp
    if datetime_cols[0] in df.columns and datetime_cols[1] in df.columns:
        try:
            df['__timestamp'] = pd.to_datetime(df[datetime_cols[0]].astype(str)) + \
                                pd.to_timedelta(df[datetime_cols[1]].astype(int), unit='h')
        except Exception:
            # fallback: if Hour is already part of Date or differently formatted
            df['__timestamp'] = pd.to_datetime(df[datetime_cols[0]].astype(str))
    else:
        # If no date/hour columns, or names differ, assume data already ordered
        df['__timestamp'] = pd.to_datetime(df.index)

    # sanitize boolean columns by converting True/False to 1/0
    bool_like = ['is_covid', 'VALID_DISPATCH_RSPNS_TIME_INDC', 'VALID_INCIDENT_RSPNS_TIME_INDC',
                 'Held_indicator','Reopen_Indicator','Special_event_indicator','STANDBY_INDICATOR',
                 'TRANSFER_INDICATOR','Transfer_Indicator']
    for c in bool_like:
        if c in df.columns:
            df[c] = df[c].astype(float).fillna(0.0)

    # Fill NA for numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # One-hot encode BOROUGH if present
    if 'BOROUGH' in df.columns:
        borough_dummies = pd.get_dummies(df['BOROUGH'].fillna("UNKNOWN"), prefix='BOROUGH')
        df = pd.concat([df.drop(columns=['BOROUGH']), borough_dummies], axis=1)

    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")

    # Sort by ZIP_CODE then timestamp (the dataset description says it's sorted, but we ensure it)
    if 'ZIP_CODE' in df.columns:
        df = df.sort_values(['ZIP_CODE', '__timestamp']).reset_index(drop=True)
    else:
        df = df.sort_values('__timestamp').reset_index(drop=True)

    return df


def create_sequences_from_df(df: pd.DataFrame,
                             seq_len: int,
                             target_col: str,
                             group_col: str = 'ZIP_CODE') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for each group (ZIP_CODE).
    For each group, create samples where X = past seq_len rows features and y = target at next timestep.

    Returns:
        X: np.array of shape (N_samples, seq_len, n_features)
        y: np.array of shape (N_samples,)
    """
    feature_cols = [c for c in df.columns if c not in [group_col, '__timestamp', target_col]]
    X_list = []
    y_list = []
    groups = df[group_col].unique() if group_col in df.columns else [None]

    for g in groups:
        if group_col in df.columns:
            sub = df[df[group_col] == g].sort_values('__timestamp').reset_index(drop=True)
        else:
            sub = df.sort_values('__timestamp').reset_index(drop=True)

        values = sub[feature_cols].values
        targets = sub[target_col].values

        # sliding window
        for i in range(len(sub) - seq_len):
            x = values[i:i + seq_len]
            y = targets[i + seq_len]  # next hour
            X_list.append(x)
            y_list.append(y)

    X = np.stack(X_list, axis=0)  # (N, seq_len, n_features)
    y = np.array(y_list, dtype=np.float32)
    return X, y, feature_cols


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------
# Models
# -----------------------

class MLPModel(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        inp = seq_len * n_features
        layers = []
        prev = inp
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        b = x.size(0)
        x = x.view(b, -1)
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size=128, num_layers=2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=0.1 if num_layers > 1 else 0.0)
        direction = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (hn, cn) = self.lstm(x)  # out: (batch, seq_len, hidden)
        # use last time-step
        last = out[:, -1, :]
        return self.fc(last)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        return self.bn(self.relu(self.conv(x)))


class TCNModel(nn.Module):
    def __init__(self, n_features, num_channels=[64, 64], kernel_size=3):
        super().__init__()
        layers = []
        in_ch = n_features
        seq_layers = []
        dilation = 1
        for ch in num_channels:
            padding = (kernel_size - 1) * dilation
            seq_layers.append(TCNBlock(in_ch, ch, kernel_size, dilation, padding))
            in_ch = ch
            dilation *= 2
        self.tcn = nn.Sequential(*seq_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> convert to (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.input_fc = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_fc(x)
        x = self.pos_enc(x)
        out = self.transformer(x)  # (batch, seq_len, d_model)
        last = out[:, -1, :]  # use last token representation
        return self.fc_out(last)

# -----------------------
# Training Loop
# -----------------------

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
            preds_list.append(preds.cpu().numpy())
            trues_list.append(yb.cpu().numpy())
    preds = np.vstack(preds_list).squeeze()
    trues = np.vstack(trues_list).squeeze()
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    return total_loss / len(loader.dataset), mae, rmse


def run_training(model, train_loader, val_loader, lr=1e-3, epochs=30, weight_decay=0.0, device='cpu'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d}  TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  ValMAE={val_mae:.4f}  ValRMSE={val_rmse:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -----------------------
# Main: data loading, preprocessing, model runs
# -----------------------

def main(args):
    # Load CSV
    df = pd.read_csv(args.data_path)

    # Preprocess
    df = parse_and_preprocess(df, target_col=args.target_col)

    # Train/val split at ZIP_CODE level to preserve continuity
    if 'ZIP_CODE' in df.columns:
        zip_codes = df['ZIP_CODE'].unique()
        train_zips, val_zips = train_test_split(zip_codes, test_size=args.val_ratio, random_state=42)
        df_train = df[df['ZIP_CODE'].isin(train_zips)].reset_index(drop=True)
        df_val = df[df['ZIP_CODE'].isin(val_zips)].reset_index(drop=True)
    else:
        # If no zip column, split by time (last X% as val)
        cutoff = int(len(df) * (1 - args.val_ratio))
        df_train = df.iloc[:cutoff].reset_index(drop=True)
        df_val = df.iloc[cutoff:].reset_index(drop=True)

    # Fit scaler on training numeric features (exclude group cols and timestamp and target)
    exclude_cols = ['ZIP_CODE', '__timestamp', args.target_col]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    # apply scaler
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_train_scaled[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_val_scaled[feature_cols] = scaler.transform(df_val[feature_cols].values)

    # Create sequences using the same seq_len
    seq_len = args.seq_len

    X_train, y_train, feat_cols = create_sequences_from_df(df_train_scaled, seq_len, args.target_col, group_col='ZIP_CODE' if 'ZIP_CODE' in df.columns else None)
    X_val, y_val, _ = create_sequences_from_df(df_val_scaled, seq_len, args.target_col, group_col='ZIP_CODE' if 'ZIP_CODE' in df.columns else None)

    print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    n_features = X_train.shape[2]

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 1) MLP
    print("\n--- Training MLP ---")
    mlp = MLPModel(seq_len=seq_len, n_features=n_features, hidden_dims=[256, 128])
    mlp = run_training(mlp, train_loader, val_loader, lr=1e-3, epochs=args.epochs, device=device)

    # 2) LSTM
    print("\n--- Training LSTM ---")
    lstm = LSTMModel(n_features=n_features, hidden_size=128, num_layers=2, bidirectional=False)
    lstm = run_training(lstm, train_loader, val_loader, lr=1e-3, epochs=args.epochs, device=device)

    # 3) TCN
    print("\n--- Training TCN ---")
    tcn = TCNModel(n_features=n_features, num_channels=[64, 64], kernel_size=3)
    tcn = run_training(tcn, train_loader, val_loader, lr=1e-3, epochs=args.epochs, device=device)

    # 4) Transformer
    print("\n--- Training Transformer ---")
    transformer = TransformerModel(n_features=n_features, d_model=128, nhead=4, num_layers=2, dim_feedforward=256)
    transformer = run_training(transformer, train_loader, val_loader, lr=1e-3, epochs=args.epochs, device=device)

    # Evaluate final models on validation and print metrics
    from pprint import pprint
    models = {'MLP': mlp, 'LSTM': lstm, 'TCN': tcn, 'Transformer': transformer}
    results = {}
    for name, model in models.items():
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, nn.MSELoss(), device)
        results[name] = {'val_loss': val_loss, 'val_mae': val_mae, 'val_rmse': val_rmse}
    print("\n=== Validation Results ===")
    pprint(results)

    # Optionally save models
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for name, model in models.items():
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{name}_model.pt"))
        print("Saved models to", args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target_col", type=str, default="call count", help="Target column name for hourly call count")
    parser.add_argument("--seq_len", type=int, default=24, help="Sequence length (default 24 hours)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save final model weights")
    args = parser.parse_args()
    main(args)

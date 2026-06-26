"""Deep-learning volatility forecast (LSTM + Transformer) vs the HAR baseline.

Runs in the separate Python-3.12 venv (.venv-dl) that has PyTorch. Reads the
cached CSVs directly so it needs no other project deps.

Honest hypothesis: on daily data, sequence models will NOT beat a 3-term HAR
regression. We measure exactly that on a held-out test set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NAMES = ["SPY", "AAPL", "JPM", "XOM", "MSFT", "JNJ"]
L = 60          # sequence length (trading days of history)
H = 21          # forecast horizon
TD = 252
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


def build_sequences():
    seqs, ys, dates, har = [], [], [], []
    for t in NAMES:
        df = pd.read_csv(DATA / f"{t}.csv", index_col="Date", parse_dates=True)
        ret = np.log(df["Close"]).diff()
        feat = pd.DataFrame({
            "ret": ret,
            "absret": ret.abs(),
            "v5": ret.rolling(5).std() * np.sqrt(TD),
            "v21": ret.rolling(21).std() * np.sqrt(TD),
            "v63": ret.rolling(63).std() * np.sqrt(TD),
        })
        y = ret.shift(-H).rolling(H).std() * np.sqrt(TD)            # forward vol
        # HAR features at decision time t
        har_t = pd.DataFrame({
            "rv_d": ret.abs() * np.sqrt(TD),
            "rv_w": ret.rolling(5).std() * np.sqrt(TD),
            "rv_m": ret.rolling(22).std() * np.sqrt(TD),
        })
        F = feat.values
        for i in range(L, len(df) - H):
            if np.isnan(F[i - L:i]).any() or np.isnan(y.iloc[i]) or har_t.iloc[i].isna().any():
                continue
            seqs.append(F[i - L:i])
            ys.append(y.iloc[i])
            dates.append(df.index[i])
            har.append(har_t.iloc[i].values)
    X = np.asarray(seqs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    d = pd.to_datetime(dates)
    har = np.asarray(har, dtype=np.float32)
    order = np.argsort(d.values)
    return X[order], y[order], d[order], har[order]


class LSTMReg(nn.Module):
    def __init__(self, f, hid=32):
        super().__init__()
        self.lstm = nn.LSTM(f, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransReg(nn.Module):
    def __init__(self, f, d=32, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(f, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        h = self.tr(self.embed(x) + self.pos)
        return self.head(h.mean(1)).squeeze(-1)


def train(model, Xtr, ytr, Xva, yva, epochs=60, bs=128, patience=8):
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, device=DEV); ytr_t = torch.tensor(ytr, device=DEV)
    Xva_t = torch.tensor(Xva, device=DEV); yva_t = torch.tensor(yva, device=DEV)
    best, best_state, bad = 1e9, None, 0
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(n).split(bs):
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xva_t), yva_t).item()
        if v < best - 1e-7:
            best, best_state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def r2_oos(y, p, m):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - m) ** 2))


def main():
    print(f"device: {DEV}")
    X, y, d, har = build_sequences()
    print(f"sequences: {X.shape}, span {d.min().date()}..{d.max().date()}")
    n = len(X)
    i_tr, i_va = int(n * 0.7), int(n * 0.85)
    # standardise features on TRAIN stats only
    mu = X[:i_tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[:i_tr].reshape(-1, X.shape[-1]).std(0) + 1e-8
    Xs = (X - mu) / sd
    Xtr, Xva, Xte = Xs[:i_tr], Xs[i_tr:i_va], Xs[i_va:]
    ytr, yva, yte = y[:i_tr], y[i_tr:i_va], y[i_va:]
    ymean = ytr.mean()

    rows = []
    # HAR baseline on identical split
    lr = LinearRegression().fit(har[:i_tr], y[:i_tr])
    p = lr.predict(har[i_va:])
    rows.append(("HAR (OLS baseline)", r2_oos(yte, p, ymean),
                 float(np.sqrt(np.mean((yte - p) ** 2)))))

    for name, M in [("LSTM", LSTMReg(X.shape[-1])), ("Transformer", TransReg(X.shape[-1]))]:
        M = train(M, Xtr, ytr, Xva, yva)
        M.eval()
        with torch.no_grad():
            p = M(torch.tensor(Xte, device=DEV)).cpu().numpy()
        rows.append((name, r2_oos(yte, p, ymean),
                     float(np.sqrt(np.mean((yte - p) ** 2)))))
        print(f"  {name:12} R2_oos={rows[-1][1]:.3f}")

    res = pd.DataFrame(rows, columns=["model", "R2_oos", "RMSE"]).set_index("model")
    out = ROOT / "reports" / "deep_learning.md"
    body = "\n".join("| " + " | ".join([i, f"{r.R2_oos:.4f}", f"{r.RMSE:.4f}"]) + " |"
                     for i, r in res.iterrows())
    out.write_text(
        "# Deep learning vs HAR — volatility (next-21d realised)\n\n"
        f"*Pooled sequences:* {len(NAMES)} names, window L={L}d, "
        f"{n} samples, chronological 70/15/15 split. Device: {DEV}.\n\n"
        "| model | R2_oos | RMSE |\n| --- | --- | --- |\n" + body +
        "\n\n**Verdict:** if the LSTM/Transformer do not clearly exceed the "
        "3-parameter HAR regression, the honest conclusion is that sequence-model "
        "complexity buys nothing here — classical econometrics is the right tool, "
        "and the website's vol engine should use HAR / forecast-combination.\n"
    )
    print(res.round(4).to_string())
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import yfinance as yf

# ── cargar datos ──────────────────────────────────────────────────────────────
y_te     = np.load("y_te.npy")
bnn_mean = np.load("bnn_mean.npy")
bnn_epi  = np.load("bnn_epi.npy")
garch    = np.load("garch_vol.npy")

meta     = json.load(open("meta.json"))
dates_te = np.array(meta["dates_te"])
dates_dt = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates_te])
t_full   = dates_dt

# ── retornos reales ───────────────────────────────────────────────────────────
df      = yf.download(meta["ticker"], start=meta["start"], end=meta["end"],
                      auto_adjust=True, progress=False)
prices  = df["Close"].squeeze()
log_ret = np.log(prices / prices.shift(1)).dropna()
ret_te  = log_ret.loc[log_ret.index.strftime("%Y-%m-%d").isin(dates_te)].values
ret_te  = ret_te[:len(y_te)]

# ── estrategias ───────────────────────────────────────────────────────────────
bnn_vol   = np.exp(0.5 * bnn_mean)
garch_vol = np.exp(0.5 * garch)
TARGET_VOL = bnn_vol.mean()
W_MIN, W_MAX = 0.2, 2.0

def vol_strategy(vol_pred, returns):
    w = np.clip(TARGET_VOL / vol_pred, W_MIN, W_MAX)
    return w * returns

def vol_strategy_bnn(vol_pred, uncertainty, returns):
    pen = 1.0 / (1.0 + uncertainty / uncertainty.mean())
    w   = np.clip((TARGET_VOL / vol_pred) * pen, W_MIN, W_MAX)
    return w * returns

r_bnn   = vol_strategy_bnn(bnn_vol, bnn_epi, ret_te)
r_garch = vol_strategy(garch_vol, ret_te)
r_naive = ret_te

# ── métricas ──────────────────────────────────────────────────────────────────
def sharpe(r, ann=252):
    return np.sqrt(ann) * r.mean() / (r.std() + 1e-8)

def max_drawdown(W):
    peak = np.maximum.accumulate(W)
    return ((W - peak) / peak).min()

def annualized_return(W):
    n_years = max(len(W) / 252, 1e-6)
    return (W[-1] ** (1 / n_years)) - 1

# ── subperiodos desde CONFIG ──────────────────────────────────────────────────
from config import CONFIG
subperiodos = CONFIG["subperiodos"]

n_sub = len(subperiodos)
fig, axes = plt.subplots(n_sub, 2, figsize=(16, 4 * n_sub))
fig.patch.set_facecolor("white")

if n_sub == 1:
    axes = [axes]

for row, (nombre, (ini, fin)) in enumerate(subperiodos.items()):
    ini_dt = datetime.strptime(ini, "%Y-%m-%d")
    fin_dt = datetime.strptime(fin, "%Y-%m-%d")
    mask   = (dates_dt >= ini_dt) & (dates_dt <= fin_dt)

    if mask.sum() == 0:
        for col in range(2):
            axes[row][col].text(0.5, 0.5, f"{nombre}\n(fuera del test set)",
                                ha="center", va="center",
                                transform=axes[row][col].transAxes,
                                fontsize=10, color="#888888")
            axes[row][col].axis("off")
        continue

    t_sub   = dates_dt[mask]
    r_b_sub = r_bnn[mask]
    r_g_sub = r_garch[mask]
    r_n_sub = r_naive[mask]

    W_bnn   = np.exp(np.cumsum(r_b_sub))
    W_garch = np.exp(np.cumsum(r_g_sub))
    W_naive = np.exp(np.cumsum(r_n_sub))

    # ── panel izquierdo: riqueza acumulada ────────────────────────────────────
    ax1 = axes[row][0]
    ax1.set_facecolor("white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#dddddd")

    ax1.plot(t_sub, W_naive, color="#333333", lw=1.4, label="Buy & Hold")
    ax1.plot(t_sub, W_garch, color="#D85A30", lw=1.5, ls="--", label="GARCH")
    ax1.plot(t_sub, W_bnn,   color="#1D9E75", lw=2.0, label="BNN + σ_epi")

    ax1.set_title(f"{nombre} — riqueza acumulada", fontsize=10, color="#111111")
    ax1.set_ylabel("Riqueza", fontsize=9, color="#333333")
    ax1.legend(facecolor="white", edgecolor="#dddddd", fontsize=8)
    ax1.grid(axis="y", color="#eeeeee", lw=0.8)
    ax1.tick_params(colors="#555555", labelsize=8)

    # métricas en el título
    for W, name in [(W_bnn, "BNN"), (W_garch, "GARCH"), (W_naive, "B&H")]:
        print(f"{nombre} | {name:6s}  "
              f"ret={annualized_return(W)*100:.1f}%  "
              f"sharpe={sharpe(np.diff(np.log(W), prepend=np.log(W[0]))):.2f}  "
              f"maxdd={max_drawdown(W)*100:.1f}%")

    # ── panel derecho: drawdown ───────────────────────────────────────────────
    ax2 = axes[row][1]
    ax2.set_facecolor("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#dddddd")

    for W, color, label, ls in [
        (W_bnn,   "#1D9E75", "BNN + σ_epi", "-"),
        (W_garch, "#D85A30", "GARCH",        "--"),
        (W_naive, "#333333", "Buy & Hold",   "-"),
    ]:
        peak = np.maximum.accumulate(W)
        dd   = (W - peak) / peak
        ax2.plot(t_sub, dd * 100, color=color, lw=1.2, ls=ls, label=label)
        ax2.fill_between(t_sub, dd * 100, 0, color=color, alpha=0.08)

    ax2.set_title(f"{nombre} — drawdown", fontsize=10, color="#111111")
    ax2.set_ylabel("Drawdown (%)", fontsize=9, color="#333333")
    ax2.legend(facecolor="white", edgecolor="#dddddd", fontsize=8)
    ax2.grid(axis="y", color="#eeeeee", lw=0.8)
    ax2.tick_params(colors="#555555", labelsize=8)

print()
plt.xticks(color="#555555", fontsize=8)
plt.tight_layout(h_pad=1.0)
plt.savefig("subperiodos.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
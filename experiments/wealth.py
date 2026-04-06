import numpy as np
import matplotlib.pyplot as plt
import json, sys, os
from datetime import datetime
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models'))
from config import CONFIG

# ── cargar datos ──────────────────────────────────────────────────────────────
OUT_DIR  = f"results/{CONFIG['ticker'].replace('^', '').replace('=', '_')}"
meta     = json.load(open(f"{OUT_DIR}/meta.json"))
dates    = np.array(meta["dates_te"])
t        = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates])

READ_DIR = OUT_DIR + "/data"
SAVE_DIR = OUT_DIR + "/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

y_te     = np.load(f"{READ_DIR}/y_te.npy")
bnn_mean = np.load(f"{READ_DIR}/bnn_mean.npy")
bnn_epi  = np.load(f"{READ_DIR}/bnn_epi.npy")
bnn_alea  = np.load(f"{READ_DIR}/bnn_alea.npy")
bnn_tot  = np.load(f"{READ_DIR}/bnn_total.npy")
garch    = np.load(f"{READ_DIR}/garch_vol.npy")
bnn_mean_tr = np.load(f"{READ_DIR}/bnn_mean_tr.npy")
bnn_epi_tr = np.load(f"{READ_DIR}/bnn_epi_tr.npy")

# ── retornos reales ───────────────────────────────────────────────────────────
df      = yf.download(meta["ticker"], start=meta["start"], end=meta["end"],
                      auto_adjust=True, progress=False)
log_ret = np.log(df["Close"].squeeze() / df["Close"].squeeze().shift(1)).dropna()
ret_te  = log_ret.loc[log_ret.index.strftime("%Y-%m-%d").isin(dates)].values
ret_te  = ret_te[:len(y_te)]

# ── escala de volatilidad con corrección de Jensen ────────────────────────────
bnn_vol           = np.sqrt(bnn_mean)   # sqrt(E[r²|x,D])
bnn_epi_corrected = bnn_epi             # ya en escala r²
garch_vol = np.exp(0.5 * garch)

# nivel base epistémica desde train

# ── métricas ──────────────────────────────────────────────────────────────────
def sharpe(r, ann=252):
    return np.sqrt(ann) * r.mean() / (r.std() + 1e-8)

def max_drawdown(W):
    peak = np.maximum.accumulate(W)
    return ((W - peak) / peak).min()

def annualized_return(W):
    return (W[-1] ** (252 / len(W))) - 1

# ── estrategias ───────────────────────────────────────────────────────────────
TARGET_VOL_GARCH = float(meta["garch_target_vol"])
TARGET_VOL = float(np.sqrt(bnn_mean_tr).mean())
W_MIN, W_MAX = 0.2, 2.0

def vol_strategy(vol_pred, returns):
    w = np.clip(TARGET_VOL_GARCH / vol_pred, W_MIN, W_MAX)
    return w * returns

def vol_strategy_bnn(vol_pred, uncertainty, returns):
    pen = 1.0 / (1.0 + uncertainty / bnn_epi_tr.mean())
    w   = np.clip((TARGET_VOL / vol_pred) * pen, W_MIN, W_MAX)
    return w * returns

def vol_strategy_bnn_compo(vol_pred, uncertainty, returns, n_std=0):
    w = np.clip(TARGET_VOL / vol_pred, W_MIN, W_MAX).copy()
    for i in range(1, len(uncertainty)):
        mu  = uncertainty[:i].mean()
        std = uncertainty[:i].std() if i > 1 else 0
        thr = mu + n_std * std
        if uncertainty[i] > thr:
            excess  = uncertainty[i] - thr
            w[i]   *= 0
    return w * returns

r_bnn_pen   = vol_strategy_bnn(bnn_vol, bnn_epi_corrected, ret_te)
r_bnn_compo = vol_strategy_bnn_compo(bnn_vol, bnn_epi_corrected, ret_te)
r_garch     = vol_strategy(garch_vol, ret_te)
r_naive     = ret_te

W_bnn_pen   = np.exp(np.cumsum(r_bnn_pen))
W_bnn_compo = np.exp(np.cumsum(r_bnn_compo))
W_garch     = np.exp(np.cumsum(r_garch))
W_naive     = np.exp(np.cumsum(r_naive))

print(f"{'Modelo':20s}  {'Ret. anual':>10}  {'Sharpe':>8}  {'Max DD':>8}  {'Riqueza':>8}")
print("─" * 65)
for name, W, r in [
    ("BNN threshold",     W_bnn_compo, r_bnn_compo),
    ("GARCH",             W_garch,     r_garch),
    ("Buy & Hold",        W_naive,     r_naive),
]:
    print(f"{name:20s}  {annualized_return(W)*100:>9.2f}%  "
          f"{sharpe(r):>8.3f}  {max_drawdown(W)*100:>7.2f}%  "
          f"{W[-1]:>8.3f}")

# ── paleta ────────────────────────────────────────────────────────────────────
C_BNN_PEN   = "#1a5fa8"
C_BNN_COMPO = "#1b4332"
C_GARCH     = "#D85A30"
C_HOLD      = "#aaaaaa"

def setup_ax(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)

# ── gráfico 1: riqueza acumulada ──────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 5))
fig1.patch.set_facecolor("white")
setup_ax(ax1)

ax1.plot(t, W_naive,     color=C_HOLD,      lw=1.4,        label="Buy & Hold")
ax1.plot(t, W_garch,     color=C_GARCH,     lw=1.5, ls="--", label="GARCH")
# ax1.plot(t, W_bnn_pen,   color=C_BNN_PEN,   lw=1.5, ls="-.", label="BNN pen. continua")
ax1.plot(t, W_bnn_compo, color=C_BNN_PEN, lw=2.0,        label="Red neuronal bayesiana")

ax1.set_ylabel("Riqueza acumulada", fontsize=10, color="#333333")
ax1.legend(facecolor="white", edgecolor="#dddddd", fontsize=9, loc="upper left")
plt.xticks(color="#555555", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/plot_wealth.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

# # ── gráfico 2: drawdown ───────────────────────────────────────────────────────
# fig2, ax2 = plt.subplots(figsize=(14, 4))
# fig2.patch.set_facecolor("white")
# setup_ax(ax2)

# for W, color, label, ls in [
#     (W_bnn_compo, C_BNN_COMPO, "BNN threshold",    "-"),
#     (W_bnn_pen,   C_BNN_PEN,   "BNN pen. continua", "-."),
#     (W_garch,     C_GARCH,     "GARCH",              "--"),
#     (W_naive,     C_HOLD,      "Buy & Hold",         "-"),
# ]:
#     peak = np.maximum.accumulate(W)
#     dd   = (W - peak) / peak
#     ax2.plot(t, dd * 100, color=color, lw=1.2, ls=ls, label=label)
#     ax2.fill_between(t, dd * 100, 0, color=color, alpha=0.06)

# ax2.set_ylabel("Drawdown (%)", fontsize=10, color="#333333")
# ax2.legend(facecolor="white", edgecolor="#dddddd", fontsize=9, loc="lower left")
# plt.xticks(color="#555555", fontsize=9)
# plt.tight_layout()
# plt.savefig(f"{SAVE_DIR}/plot_drawdown.png", dpi=150, bbox_inches="tight", facecolor="white")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import yfinance as yf

y_te     = np.load("y_te.npy")
bnn_mean = np.load("bnn_mean.npy")
bnn_epi  = np.load("bnn_epi.npy")
garch    = np.load("garch_vol.npy")

meta    = json.load(open("meta.json"))
ticker  = meta["ticker"]
start   = meta["start"]
end     = meta["end"]
dates   = np.array(meta["dates_te"])   # ← mover aquí, antes del download
t     = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates])  # ← añadir

df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

# ── retornos reales ───────────────────────────────────────────────────────────
prices  = df["Close"].squeeze()
log_ret = np.log(prices / prices.shift(1)).dropna()
ret_te  = log_ret.loc[log_ret.index.strftime("%Y-%m-%d").isin(dates)].values
ret_te  = ret_te[:len(y_te)]

# ── convertir varianza a volatilidad para el scaling ─────────────────────────
bnn_vol   = np.exp(0.5 * bnn_mean)
garch_vol = np.exp(0.5 * garch)

# ── métricas ──────────────────────────────────────────────────────────────────
def sharpe(r, ann=252):
    return np.sqrt(ann) * r.mean() / r.std()

def max_drawdown(W):
    peak = np.maximum.accumulate(W)
    return ((W - peak) / peak).min()

def annualized_return(W):
    n_years = len(W) / 252
    return (W[-1] ** (1 / n_years)) - 1

# ── estrategias ───────────────────────────────────────────────────────────────
   # o np.exp(0.5 * y_mean) si tienes y_mean en log-escala
W_MIN, W_MAX = 0.2, 2.0
TARGET_VOL = bnn_vol.mean()

def vol_strategy(vol_pred, returns):
    w = np.clip(TARGET_VOL / vol_pred, W_MIN, W_MAX)
    return w * returns

def vol_strategy_bnn(vol_pred, uncertainty, returns):
    pen  = 1.0 / (1.0 + uncertainty / uncertainty.mean())
    w = np.clip((TARGET_VOL / vol_pred) * pen, W_MIN, W_MAX)
    return w * returns

lambda_unc = bnn_vol.mean() / bnn_epi.mean()

r_bnn_naive = vol_strategy(bnn_vol, ret_te)
r_bnn_unc   = vol_strategy_bnn(bnn_vol, bnn_epi, ret_te)
r_garch     = vol_strategy(garch_vol, ret_te)
r_naive     = ret_te

W_bnn_naive = np.exp(np.cumsum(r_bnn_naive))
W_bnn_unc   = np.exp(np.cumsum(r_bnn_unc))
W_garch     = np.exp(np.cumsum(r_garch))
W_naive     = np.exp(np.cumsum(r_naive))

for name, W, r in [
    ("BNN sin σ_epi", W_bnn_naive, r_bnn_naive),
    ("BNN con σ_epi", W_bnn_unc,   r_bnn_unc),
    ("GARCH",         W_garch,     r_garch),
    ("Buy&Hold",      W_naive,     r_naive),
]:
    print(f"{name:15s}  retorno anual={annualized_return(W)*100:.2f}%  "
          f"sharpe={sharpe(r):.3f}  max_dd={max_drawdown(W)*100:.2f}%  "
          f"riqueza_final={W[-1]:.3f}")

# ── figura ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")

ax = axes[0]
ax.plot(t[:len(W_naive)],     W_naive,     color="#333333", lw=1.4, label="Buy & Hold")
ax.plot(t[:len(W_garch)],     W_garch,     color="#D85A30", lw=1.5, ls="--", label="GARCH vol-scaling")
ax.plot(t[:len(W_bnn_naive)], W_bnn_naive, color="#5bc8a0", lw=1.5, ls=":",  label="BNN vol-scaling")
ax.plot(t[:len(W_bnn_unc)],   W_bnn_unc,   color="#1D9E75", lw=2.2, label="BNN vol-scaling + σ epistémica")

ax.set_ylabel("Riqueza acumulada", fontsize=10, color="#333333")
ax.set_title("Riqueza acumulada — estrategia vol-scaling (test set)",
             fontsize=12, pad=10, color="#111111")
ax.legend(facecolor="white", edgecolor="#dddddd", fontsize=9, loc="upper left")
ax.grid(axis="y", color="#eeeeee", lw=0.8)
ax.tick_params(colors="#555555")

ax2 = axes[1]
for W, color, label, ls in [
    (W_bnn_unc,   "#1D9E75", "BNN + σ_epi", "-"),
    (W_bnn_naive, "#5bc8a0", "BNN sin σ_epi", ":"),
    (W_garch,     "#D85A30", "GARCH", "--"),
    (W_naive,     "#333333", "Buy&Hold", "-"),
]:
    peak = np.maximum.accumulate(W)
    dd   = (W - peak) / peak
    ax2.plot(t[:len(dd)], dd * 100, color=color, lw=1.2, ls=ls, label=label)
    ax2.fill_between(t[:len(dd)], dd * 100, 0, color=color, alpha=0.06)

ax2.set_ylabel("Drawdown (%)", fontsize=10, color="#333333")
ax2.tick_params(colors="#555555")
ax2.legend(facecolor="white", edgecolor="#dddddd", fontsize=9, loc="lower left")
ax2.grid(axis="y", color="#eeeeee", lw=0.8)

plt.xticks(color="#555555", fontsize=9)
plt.tight_layout(h_pad=0.4)
plt.savefig("wealth_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

print("Exposure BNN:", np.mean(np.abs(r_bnn_unc)) / np.mean(np.abs(ret_te)))
print("Exposure GARCH:", np.mean(np.abs(r_garch)) / np.mean(np.abs(ret_te)))
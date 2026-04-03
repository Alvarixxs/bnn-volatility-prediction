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
dates_te = np.array(meta["dates_te"])
dates_dt = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates_te])

READ_DIR = OUT_DIR + "/data"
SAVE_DIR = OUT_DIR + "/plots/period"
os.makedirs(SAVE_DIR, exist_ok=True)

y_te     = np.load(f"{READ_DIR}/y_te.npy")
bnn_mean = np.load(f"{READ_DIR}/bnn_mean.npy")
bnn_epi  = np.load(f"{READ_DIR}/bnn_epi.npy")
bnn_tot  = np.load(f"{READ_DIR}/bnn_total.npy")
garch    = np.load(f"{READ_DIR}/garch_vol.npy")

# ── retornos reales ───────────────────────────────────────────────────────────
df      = yf.download(meta["ticker"], start=meta["start"], end=meta["end"],
                      auto_adjust=True, progress=False)
log_ret = np.log(df["Close"].squeeze() / df["Close"].squeeze().shift(1)).dropna()
ret_te  = log_ret.loc[log_ret.index.strftime("%Y-%m-%d").isin(dates_te)].values
ret_te  = ret_te[:len(y_te)]

# ── corrección de Jensen ──────────────────────────────────────────────────────
bnn_var_correction = 0.5 * (bnn_tot ** 2)
bnn_vol            = np.exp(0.5 * (bnn_mean + bnn_var_correction))
bnn_epi_corrected  = np.exp(0.5 * (bnn_mean + bnn_var_correction + bnn_epi)) - bnn_vol
garch_vol          = np.exp(0.5 * garch)

# ── estrategias ───────────────────────────────────────────────────────────────
TARGET_VOL = bnn_vol.mean()
W_MIN, W_MAX = 0.2, 2.0

def vol_strategy(vol_pred, returns):
    w = np.clip(TARGET_VOL / vol_pred, W_MIN, W_MAX)
    return w * returns

def vol_strategy_bnn(vol_pred, uncertainty, returns):
    pen = 1.0 / (1.0 + uncertainty / uncertainty.mean())
    w   = np.clip((TARGET_VOL / vol_pred) * pen, W_MIN, W_MAX)
    return w * returns

r_bnn   = vol_strategy_bnn(bnn_vol, bnn_epi_corrected, ret_te)
r_garch = vol_strategy(garch_vol, ret_te)
r_naive = ret_te

# ── métricas ──────────────────────────────────────────────────────────────────
def sharpe(r, ann=252):
    return np.sqrt(ann) * r.mean() / (r.std() + 1e-8)

def max_drawdown(W):
    peak = np.maximum.accumulate(W)
    return ((W - peak) / peak).min()

def annualized_return(W):
    return (W[-1] ** (252 / max(len(W), 1))) - 1

# ── paleta ────────────────────────────────────────────────────────────────────
C_BNN   = "#1a5fa8"
C_GARCH = "#D85A30"
C_HOLD  = "#333333"

def setup_ax(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
    ax.tick_params(colors="#555555", labelsize=8)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)

def format_dates(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0,
             color="#555555", fontsize=9)

# ── gráfico por subperiodo ────────────────────────────────────────────────────
for nombre, (ini, fin) in CONFIG["subperiodos"].items():
    ini_dt = datetime.strptime(ini, "%Y-%m-%d")
    fin_dt = datetime.strptime(fin, "%Y-%m-%d")
    mask   = (dates_dt >= ini_dt) & (dates_dt <= fin_dt)

    if mask.sum() == 0:
        print(f"{nombre}: fuera del test set, omitido.")
        continue

    t_sub   = dates_dt[mask]
    W_bnn   = np.exp(np.cumsum(r_bnn[mask]))
    W_garch = np.exp(np.cumsum(r_garch[mask]))
    W_naive = np.exp(np.cumsum(r_naive[mask]))

    print(f"\n── {nombre} ──")
    for name, W, r in [("BNN",        W_bnn,   r_bnn[mask]),
                        ("GARCH",      W_garch, r_garch[mask]),
                        ("Buy & Hold", W_naive, r_naive[mask])]:
        print(f"  {name:12s}  ret={annualized_return(W)*100:.1f}%  "
              f"sharpe={sharpe(r):.2f}  maxdd={max_drawdown(W)*100:.1f}%")

    fname = nombre.replace(" ", "_").replace("/", "-")

    # ── riqueza acumulada ─────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    fig1.patch.set_facecolor("white")
    setup_ax(ax1)

    ax1.plot(t_sub, W_naive, color=C_HOLD,  lw=1.4, label="Buy & Hold")
    ax1.plot(t_sub, W_garch, color=C_GARCH, lw=1.5, ls="--", label="GARCH")
    ax1.plot(t_sub, W_bnn,   color=C_BNN,   lw=2.0, label="BNN")
    ax1.set_ylabel("Riqueza acumulada", fontsize=9, color="#333333")
    ax1.legend(facecolor="white", edgecolor="#dddddd", fontsize=8, loc="upper left")
    format_dates(ax1)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/plot_sub_{fname}_wealth.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.show()

    # ── drawdown ──────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    fig2.patch.set_facecolor("white")
    setup_ax(ax2)

    for W, color, label, ls in [
        (W_bnn,   C_BNN,   "BNN",        "-"),
        (W_garch, C_GARCH, "GARCH",      "--"),
        (W_naive, C_HOLD,  "Buy & Hold", "-"),
    ]:
        peak = np.maximum.accumulate(W)
        dd   = (W - peak) / peak
        ax2.plot(t_sub, dd * 100, color=color, lw=1.2, ls=ls, label=label)
        ax2.fill_between(t_sub, dd * 100, 0, color=color, alpha=0.08)

    ax2.set_ylabel("Drawdown (%)", fontsize=9, color="#333333")
    ax2.legend(facecolor="white", edgecolor="#dddddd", fontsize=8, loc="lower left")
    format_dates(ax2)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/plot_sub_{fname}_drawdown.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.show()
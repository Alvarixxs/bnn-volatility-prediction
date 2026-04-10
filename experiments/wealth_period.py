import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import json, sys, os
from datetime import datetime
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models'))
from config import CONFIG

# ── cargar datos ──────────────────────────────────────────────────────────────
OUT_DIR  = f"results/{CONFIG['ticker'].replace('^', '').replace('=', '_')}"
READ_DIR = f"{OUT_DIR}/data"
SAVE_DIR = f"{OUT_DIR}/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

meta     = json.load(open(f"{OUT_DIR}/meta.json"))
dates_te = np.array(meta["dates_te"])
dates_dt = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates_te])

y_te     = np.load(f"{READ_DIR}/y_te.npy")
bnn_mean = np.load(f"{READ_DIR}/bnn_mean.npy")
bnn_epi  = np.load(f"{READ_DIR}/bnn_epi.npy")
garch    = np.load(f"{READ_DIR}/garch_vol.npy")

# ── retornos reales ───────────────────────────────────────────────────────────
df      = yf.download(meta["ticker"], start=meta["start"], end=meta["end"],
                      auto_adjust=True, progress=False)
log_ret = np.log(df["Close"].squeeze() / df["Close"].squeeze().shift(1)).dropna()
ret_te  = log_ret.loc[log_ret.index.strftime("%Y-%m-%d").isin(dates_te)].values[:len(y_te)]

# ── volatilidad ───────────────────────────────────────────────────────────────
bnn_vol  = np.sqrt(bnn_mean)
garch_vol = np.exp(0.5 * garch)

TARGET_VOL       = bnn_vol.mean()
TARGET_VOL_GARCH = float(meta["garch_target_vol"])
W_MIN, W_MAX     = 0.2, 2.0

# ── estrategias ───────────────────────────────────────────────────────────────
def vol_strategy(vol_pred, target, returns):
    return np.clip(target / vol_pred, W_MIN, W_MAX) * returns

def vol_strategy_bnn(vol_pred, uncertainty, returns, n_std=0):
    w = np.clip(TARGET_VOL / vol_pred, W_MIN, W_MAX).copy()
    for i in range(1, len(uncertainty)):
        mu  = uncertainty[:i].mean()
        std = uncertainty[:i].std() if i > 1 else 0
        if uncertainty[i] > mu + n_std * std:
            w[i] = 0
    return w * returns

r_bnn   = vol_strategy_bnn(bnn_vol, bnn_epi, ret_te)
r_garch = vol_strategy(garch_vol, TARGET_VOL_GARCH, ret_te)
r_naive = ret_te

# ── métricas ──────────────────────────────────────────────────────────────────
def sharpe(r, ann=252):
    return np.sqrt(ann) * r.mean() / (r.std() + 1e-8)

def max_drawdown(W):
    return ((W - np.maximum.accumulate(W)) / np.maximum.accumulate(W)).min()

def annualized_return(W):
    return (W[-1] ** (252 / max(len(W), 1))) - 1

# ── estilo ────────────────────────────────────────────────────────────────────
C_BNN   = "#1a5fa8"
C_GARCH = "#D85A30"
C_HOLD  = "#aaaaaa"

def setup_ax(ax):
    ax.set_facecolor("white")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16, labelleft=False)
    ax.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def format_dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=16)

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

    print(f"── {nombre} ──")
    for name, W, r in [("Red neuronal bayesiana", W_bnn,   r_bnn[mask]),
                        ("GARCH",                  W_garch, r_garch[mask]),
                        ("Buy & Hold",             W_naive, r_naive[mask])]:
        print(f"  {name:25s}  ret={annualized_return(W)*100:.1f}%  "
              f"sharpe={sharpe(r):.2f}  maxdd={max_drawdown(W)*100:.1f}%")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    setup_ax(ax)

    ax.plot(t_sub, W_naive, color=C_HOLD,  lw=1.5, label="Buy & Hold")
    ax.plot(t_sub, W_garch, color=C_GARCH, lw=1.5, label="GARCH")
    ax.plot(t_sub, W_bnn,   color=C_BNN,   lw=1.5, label="Red neuronal bayesiana")

    ax.set_ylabel("Riqueza", fontsize=16)
    ax.legend(facecolor="white", edgecolor="#dddddd", fontsize=16, loc="lower left")
    format_dates(ax)
    plt.tight_layout()

    fname = f"{SAVE_DIR}/wealth_{CONFIG['ticker'].replace('^', '').replace('=', '_')}_{nombre.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nGuardada: {fname}")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json, sys, os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models'))
from config import CONFIG

# ── cargar datos ──────────────────────────────────────────────────────────────
OUT_DIR  = f"results/{CONFIG['ticker'].replace('^', '').replace('=', '_')}"
READ_DIR = f"{OUT_DIR}/data"
SAVE_DIR = f"{OUT_DIR}/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

meta = json.load(open(f"{OUT_DIR}/meta.json"))
t    = np.array([datetime.strptime(d, "%Y-%m-%d") for d in meta["dates_te"]])

y_te     = np.load(f"{READ_DIR}/y_te.npy")
bnn_mean = np.load(f"{READ_DIR}/bnn_mean.npy")
bnn_epi  = np.load(f"{READ_DIR}/bnn_epi.npy")
bnn_alea = np.load(f"{READ_DIR}/bnn_alea.npy")
garch    = np.load(f"{READ_DIR}/garch_vol.npy")

# ── escala de volatilidad ─────────────────────────────────────────────────────
y_vol     = np.exp(0.5 * y_te)
bnn_vol   = np.sqrt(bnn_mean)
garch_vol = np.exp(0.5 * garch)

# ── métricas ──────────────────────────────────────────────────────────────────
rmse_bnn   = np.sqrt(np.mean((bnn_vol   - y_vol)**2))
rmse_garch = np.sqrt(np.mean((garch_vol - y_vol)**2))

def qlike(sigma_pred, r2_real):
    var_pred = sigma_pred ** 2
    var_real = r2_real    ** 2
    return np.mean(var_real / var_pred - np.log(var_real / var_pred) - 1)

qlike_bnn   = qlike(bnn_vol,   y_vol)
qlike_garch = qlike(garch_vol, y_vol)

print(f"{'Modelo':10s}  {'RMSE':>10}  {'QLIKE':>10}")
print(f"{'Red neuronal bayesiana':10s}  {rmse_bnn:>10.5f} {qlike_bnn:>10.5f}")
print(f"{'GARCH':10s}  {rmse_garch:>10.5f} {qlike_garch:>10.5f}")

# ── paleta ────────────────────────────────────────────────────────────────────
C_BNN   = "#1a5fa8"
C_ALEA  = "#8fb2ff"
C_GARCH = "#D85A30"
C_REAL  = "#aaaaaa"

# ── suavizado ─────────────────────────────────────────────────────────────────
def smooth(x, w=5):
    out = np.convolve(x, np.ones(w) / w, mode='same')
    out[:w]  = x[:w]
    out[-w:] = x[-w:]
    return out

sl = slice(5, -5)

def setup_ax(ax):
    ax.set_facecolor("white")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16, labelleft=False)
    ax.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ── gráfico 1: predicciones ───────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(14, 5))
fig1.patch.set_facecolor("white")
setup_ax(ax)

ax.plot(t[sl], smooth(y_vol)[sl],     color=C_REAL,  lw=1.5, zorder=4, label="Volatilidad realizada")
ax.plot(t[sl], smooth(garch_vol)[sl], color=C_GARCH, lw=1.5, zorder=5, label="GARCH")
ax.plot(t[sl], smooth(bnn_vol)[sl],   color=C_BNN,   lw=1.5, zorder=6, label="Red neuronal bayesiana")

ax.set_ylabel("Volatilidad", fontsize=16)
ax.legend(facecolor="white", edgecolor="#dddddd", fontsize=16, loc="upper left")
plt.tight_layout()
fname = f"{SAVE_DIR}/vol_{CONFIG['ticker'].replace('^', '').replace('=', '_')}.png"
plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nGuardada: {fname}")

# ── gráfico 2: incertidumbre ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 4))
fig2.patch.set_facecolor("white")
setup_ax(ax2)

ax2.fill_between(t, 0, bnn_alea, color=C_ALEA, alpha=0.12)
ax2.plot(t, bnn_epi,  color=C_BNN,  lw=1.5, label="σ epistémica")
ax2.plot(t, bnn_alea, color=C_ALEA, lw=1.5, label="σ aleatórica")

ax2.set_ylabel("Desviación típica", fontsize=16)
ax2.legend(facecolor="white", edgecolor="#dddddd", fontsize=16, loc="upper left")
plt.tight_layout()
fname = f"{SAVE_DIR}/uncertainty_{CONFIG['ticker'].replace('^', '').replace('=', '_')}.png"
plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Guardada: {fname}")

import pandas as pd

# CSV volatilidad
df_vol = pd.DataFrame({
    'dia': np.arange(len(t)),
    'realizada': smooth(y_vol) * 100,
    'garch': smooth(garch_vol) * 100,
    'bnn': smooth(bnn_vol) * 100
})
df_vol.to_csv(f"{SAVE_DIR}/vol_GSPC.csv", index=False)
print(f"CSV guardado: {SAVE_DIR}/vol_GSPC.csv")

# CSV incertidumbre
df_unc = pd.DataFrame({
    'dia': np.arange(len(t)),
    'epistemica': bnn_epi * 100,
    'aleatoria': bnn_alea * 100
})
df_unc.to_csv(f"{SAVE_DIR}/uncertainty_GSPC.csv", index=False)
print(f"CSV guardado: {SAVE_DIR}/uncertainty_GSPC.csv")

for year in range(1998, 2027, 2):
    idx = next((i for i, d in enumerate(t) if d.year == year and d.month == 1), None)
    if idx is not None:
        print(f"{year}: {idx}")
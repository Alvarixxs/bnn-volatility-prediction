import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
bnn_mean = np.load(f"{READ_DIR}/bnn_mean.npy")  # E[r²|x,D]
bnn_epi  = np.load(f"{READ_DIR}/bnn_epi.npy")   # std epistémica en escala r²
bnn_alea = np.load(f"{READ_DIR}/bnn_alea.npy")  # std aleatórica en escala r²
bnn_tot  = np.load(f"{READ_DIR}/bnn_total.npy") # std total en escala r²
garch    = np.load(f"{READ_DIR}/garch_vol.npy")

# ── escala de volatilidad ─────────────────────────────────────────────────────
# bnn_mean ya es E[r²|x,D] — sqrt da la volatilidad
y_vol     = np.exp(0.5 * y_te)        # target: exp(0.5 * log r²) = |r|
bnn_vol   = np.sqrt(bnn_mean)         # sqrt(E[r²|x,D])
garch_vol = np.exp(0.5 * garch)

# ── paleta ────────────────────────────────────────────────────────────────────
C_BNN   = "#1a5fa8"
C_ALEA  = "#8fb2ffff"
C_EPI   = "#2257B9"
C_TOT   = "#4a4a8a"
C_GARCH = "#D85A30"
C_REAL  = "#aaaaaa"

# ── suavizado ─────────────────────────────────────────────────────────────────
def smooth(x, w=10):
    out = np.convolve(x, np.ones(w) / w, mode='same')
    out[:w]  = x[:w]
    out[-w:] = x[-w:]
    return out

W_s = 5
sl  = slice(W_s, -W_s)

def setup_ax(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
    ax.tick_params(colors="#555555", labelsize=16)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)

# ── gráfico 1: predicciones ───────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(14, 5))
fig1.patch.set_facecolor("white")
setup_ax(ax)

ax.plot(t[sl], smooth(y_vol,     W_s)[sl], color=C_REAL,  lw=1.2, zorder=4, label="Vol. realizada")
ax.plot(t[sl], smooth(garch_vol, W_s)[sl], color=C_GARCH, lw=1.5, ls="--", zorder=5, label="GARCH")
ax.plot(t[sl], smooth(bnn_vol,   W_s)[sl], color=C_BNN,   lw=1.5, zorder=6, label="Red neuronal bayesiana")

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles,
          facecolor="white", edgecolor="#dddddd", fontsize=16, loc="upper left")
ax.set_ylabel("Volatilidad diaria $\\sigma_t$", fontsize=16, color="#333333")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
plt.xticks(color="#555555", fontsize=16)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/plot_pred.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

# ── gráfico 2: incertidumbre ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 4))
fig2.patch.set_facecolor("white")
setup_ax(ax2)

ax2.fill_between(t, 0, bnn_epi, color=C_EPI, alpha=0.12)
ax2.plot(t, bnn_epi,  color=C_EPI,  lw=1.8, label="σ epistémica")
ax2.plot(t, bnn_alea, color=C_ALEA, lw=1.8, ls="--", label="σ aleatórica")
# ax2.plot(t, bnn_tot,  color=C_TOT,  lw=1.2, ls=":",  label="σ total")

# ax2.set_ylabel("Incertidumbre", fontsize=16, color="#333333")
ax2.legend(facecolor="white", edgecolor="#dddddd", fontsize=16, loc="upper left")
plt.xticks(color="#555555", fontsize=16)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/plot_uncertainty.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

# ── métricas ──────────────────────────────────────────────────────────────────
rmse_bnn   = np.sqrt(np.mean((bnn_vol   - y_vol)**2))
rmse_garch = np.sqrt(np.mean((garch_vol - y_vol)**2))
rmse_naive = np.sqrt(np.mean((y_vol[:-1] - y_vol[1:])**2))
corr_bnn   = np.corrcoef(bnn_vol,   y_vol)[0,1]
corr_garch = np.corrcoef(garch_vol, y_vol)[0,1]

print(f"{'Modelo':10s}  {'RMSE':>10}  {'Corr':>8}")
print(f"{'Naive':10s}  {rmse_naive:>10.5f}")
print(f"{'BNN':10s}  {rmse_bnn:>10.5f}  {corr_bnn:>8.4f}")
print(f"{'GARCH':10s}  {rmse_garch:>10.5f}  {corr_garch:>8.4f}")

print(f"garch_vol media: {garch_vol.mean():.6f}")
print(f"y_vol media:     {y_vol.mean():.6f}")
print(f"bnn_vol media:   {bnn_vol.mean():.6f}")

def qlike(sigma_pred, r2_real):
    """
    sigma_pred: volatilidad predicha (σ)
    r2_real:    varianza realizada (r²) = y_vol²
    """
    var_pred = sigma_pred ** 2
    var_real = r2_real    ** 2
    return np.mean(var_real / var_pred - np.log(var_real / var_pred) - 1)

# y_vol = |r| → y_vol² = r²
qlike_bnn   = qlike(bnn_vol,   y_vol)
qlike_garch = qlike(garch_vol, y_vol)
qlike_naive = qlike(y_vol[:-1], y_vol[1:])

print(f"{'Modelo':10s}  {'RMSE':>10}  {'Corr':>8}  {'QLIKE':>10}")
print(f"{'Naive':10s}  {rmse_naive:>10.5f}  {'—':>8}  {qlike_naive:>10.5f}")
print(f"{'BNN':10s}  {rmse_bnn:>10.5f}  {corr_bnn:>8.4f}  {qlike_bnn:>10.5f}")
print(f"{'GARCH':10s}  {rmse_garch:>10.5f}  {corr_garch:>8.4f}  {qlike_garch:>10.5f}")
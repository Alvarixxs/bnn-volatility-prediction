import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from datetime import datetime

y_te     = np.load("y_te.npy")
bnn_mean = np.load("bnn_mean.npy")
bnn_epi  = np.load("bnn_epi.npy")
bnn_alea = np.load("bnn_alea.npy")
bnn_tot  = np.load("bnn_total.npy")
garch    = np.load("garch_vol.npy")

meta  = json.load(open("meta.json"))
dates = np.array(meta["dates_te"])
t     = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates])

y_vol     = np.exp(0.5 * y_te)
bnn_vol   = np.exp(0.5 * bnn_mean)
garch_vol = np.exp(0.5 * garch)

def smooth(x, w=5):
    out = np.convolve(x, np.ones(w)/w, mode='same')
    out[:w]  = x[:w]
    out[-w:] = x[-w:]
    return out

W_s = 10

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.5]})
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")

ax = axes[0]
# bandas correctas: propagar incertidumbre a escala de volatilidad
bnn_vol_upper_epi = np.exp(0.5 * (bnn_mean + bnn_epi))
bnn_vol_lower_epi = np.exp(0.5 * (bnn_mean - bnn_epi))
bnn_vol_upper_tot = np.exp(0.5 * (bnn_mean + bnn_tot))
bnn_vol_lower_tot = np.exp(0.5 * (bnn_mean - bnn_tot))

# recortar W_s puntos en cada extremo para eliminar artefactos
sl = slice(W_s, -W_s)

ax.fill_between(t[sl],
                smooth(bnn_vol_lower_tot, W_s)[sl],
                smooth(bnn_vol_upper_tot, W_s)[sl],
                color="#1D9E75", alpha=0.10, zorder=2)
ax.fill_between(t[sl],
                smooth(bnn_vol_lower_epi, W_s)[sl],
                smooth(bnn_vol_upper_epi, W_s)[sl],
                color="#1D9E75", alpha=0.25, zorder=3)

ax.plot(t[sl], smooth(y_vol,     W_s)[sl], color="#aaaaaa", lw=1.2, zorder=4, label="Vol. realizada")
ax.plot(t[sl], smooth(garch_vol, W_s)[sl], color="#D85A30", lw=1.5, ls="--", zorder=5, label="GARCH(1,1)")
ax.plot(t[sl], smooth(bnn_vol,   W_s)[sl], color="#1D9E75", lw=2.0, zorder=6, label="BNN (media)")

patch_epi = mpatches.Patch(color="#1D9E75", alpha=0.35, label="±1σ epistémica")
patch_tot = mpatches.Patch(color="#1D9E75", alpha=0.12, label="±1σ total")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles + [patch_epi, patch_tot],
          facecolor="white", edgecolor="#dddddd", fontsize=9, loc="upper left")
ax.set_ylabel("Volatilidad diaria $\\sigma_t$", fontsize=10, color="#333333")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
ax.tick_params(colors="#555555")
ax.set_title("BNN vs GARCH(1,1) — predicción de volatilidad 1-día (test set)",
             fontsize=12, pad=10, color="#111111")
ax.grid(axis="y", color="#eeeeee", lw=0.8)

ax2 = axes[1]
ax2.plot(t, bnn_epi,  color="#1D9E75", lw=1.8, label="σ epistémica")
ax2.plot(t, bnn_alea, color="#D85A30", lw=1.8, ls="--", label="σ aleatórica")
ax2.plot(t, bnn_tot,  color="#7F77DD", lw=1.2, ls=":",  label="σ total")
ax2.fill_between(t, 0, bnn_epi, color="#1D9E75", alpha=0.10)

ax2.set_ylabel("Incertidumbre (log-var)", fontsize=10, color="#333333")
ax2.tick_params(colors="#555555")
ax2.legend(facecolor="white", edgecolor="#dddddd", fontsize=9, loc="upper left")
ax2.grid(axis="y", color="#eeeeee", lw=0.8)

plt.xticks(color="#555555", fontsize=9)
plt.tight_layout(h_pad=0.4)
plt.savefig("bnn_vs_garch.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

rmse_bnn   = np.sqrt(np.mean((bnn_vol   - y_vol)**2))
rmse_garch = np.sqrt(np.mean((garch_vol - y_vol)**2))
naive      = y_vol[:-1]
rmse_naive = np.sqrt(np.mean((naive - y_vol[1:])**2))
corr_bnn   = np.corrcoef(bnn_vol,   y_vol)[0,1]
corr_garch = np.corrcoef(garch_vol, y_vol)[0,1]

print(f"{'Modelo':10s}  RMSE       Corr")
print(f"{'Naive':10s}  {rmse_naive:.5f}")
print(f"{'BNN':10s}  {rmse_bnn:.5f}   {corr_bnn:.4f}")
print(f"{'GARCH':10s}  {rmse_garch:.5f}   {corr_garch:.4f}")
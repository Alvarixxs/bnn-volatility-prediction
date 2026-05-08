import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models'))

# ── configuración ─────────────────────────────────────────────────────────────
from config import CONFIG

TICKER   = CONFIG["ticker"]
OUT_DIR  = f"results/{TICKER.replace('^','').replace('=','_')}"
SAVE_DIR = f"{OUT_DIR}/data"
FIG_DIR  = f"{OUT_DIR}/plots"
os.makedirs(FIG_DIR, exist_ok=True)

# ── cargar ────────────────────────────────────────────────────────────────────
history = np.load(f"{SAVE_DIR}/elbo_history.npy")   # ELBO negativa por época
elbo    = history
epochs  = np.arange(1, len(elbo) + 1)

# ── figura ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(epochs, elbo,        color="#0246bb", linewidth=0.6, alpha=0.7, label="-ELBO (por época)")

ax.set_xlabel("Época", fontsize=10)
ax.set_ylabel("ELBO", fontsize=10)
ax.set_title(f"Convergencia de la ELBO — {TICKER}", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.legend(fontsize=9, framealpha=0.4)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/elbo_convergence.png", dpi=150, bbox_inches="tight")
print(f"Guardado en {FIG_DIR}/")
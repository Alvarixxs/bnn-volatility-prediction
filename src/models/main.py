import numpy as np
import torch
import json
import warnings
import os
warnings.filterwarnings("ignore")
import yfinance as yf
from config import CONFIG
from models import BNN
from train import train_bnn
from predict import predict_bnn
from arch import arch_model

OUT_DIR  = f"results/{CONFIG['ticker'].replace('^','').replace('=','_')}"
SAVE_DIR = f"{OUT_DIR}/data"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── datos ─────────────────────────────────────────────────────────────────────
df      = yf.download(CONFIG["ticker"], start=CONFIG["start"], end=CONFIG["end"],
                      auto_adjust=True, progress=False)
log_ret = np.log(df["Close"].squeeze() / df["Close"].squeeze().shift(1)).dropna()
r       = log_ret.values.astype(np.float32)
dates   = [str(d.date()) for d in log_ret.index]
log_var = np.log(r**2 + CONFIG["epsilon"])

W = CONFIG["window"]
X, y, dates_out = [], [], []
for i in range(W, len(log_var) - 1):
    X.append(log_var[i-W:i])
    y.append(log_var[i+1])
    dates_out.append(dates[i+1])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)          # log(r²) sin normalizar

# ── split ─────────────────────────────────────────────────────────────────────
n          = len(y)
n_tr       = int(n * CONFIG["train_frac"])
n_te       = n - n_tr
X_tr, y_tr = X[:n_tr], y[:n_tr]
X_te, y_te = X[n_tr:], y[n_tr:]
dates_te   = dates_out[n_tr:]

print(f"Ticker: {CONFIG['ticker']}")
print(f"Train:  {dates_out[0]}  →  {dates_out[n_tr-1]}  ({n_tr} días)")
print(f"Test:   {dates_te[0]}  →  {dates_te[-1]}  ({n_te} días)")

# ── normalización — solo inputs ───────────────────────────────────────────────
mu_x, std_x = X_tr.mean(0), X_tr.std(0) + 1e-8
X_tr_n = (X_tr - mu_x) / std_x
X_te_n = (X_te - mu_x) / std_x
# X_tr_n = X_tr
# X_te_n = X_te
# y_tr e y_te se pasan en escala original directamente

# ── entrenamiento BNN ─────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

y_mean = float(y_tr.mean())               # media empírica de log(r²) en train

print("\nEntrenando BNN...")
bnn = BNN(input_dim=W, hidden=CONFIG["hidden"],
          prior_std=CONFIG["prior_std"], y_mean=y_mean)
bnn, history = train_bnn(bnn, X_tr_n, y_tr, CONFIG)   # ← modificar
np.save(f"{SAVE_DIR}/elbo_history.npy", np.array(history))   # ← añadir

preds    = predict_bnn(bnn, X_te_n,  CONFIG["n_samples"])
preds_tr = predict_bnn(bnn, X_tr_n,  CONFIG["n_samples"])
print("BNN listo.")

# ── GARCH(1,1) rolling ────────────────────────────────────────────────────────
print("\nFitting GARCH rolling...")
r_pct      = r * 100
r_full_pct = r_pct[W:]
idx_test   = n_tr
res        = arch_model(r_full_pct[:idx_test], vol='Garch', p=1, q=1).fit(disp='off')
garch_logvar = []

for i in range(n_te):
    t = idx_test + i
    if i == 0 or i % 63 == 0:
        m   = arch_model(r_full_pct[:t], vol='Garch', p=1, q=1)
        res = m.fit(starting_values=res.params.values, disp='off', show_warning=False)
    m_update   = arch_model(r_full_pct[:t+1], vol='Garch', p=1, q=1)
    res_update = m_update.fit(starting_values=res.params.values,
                              disp='off', show_warning=False,
                              options={'maxiter': 1})
    var_daily = res_update.forecast(horizon=1, reindex=False)\
                           .variance.values[-1, 0] / 100**2
    garch_logvar.append(np.log(var_daily + CONFIG["epsilon"]))

garch_logvar     = np.array(garch_logvar, dtype=np.float32)
garch_train_vol  = res.conditional_volatility / 100
garch_target_vol = float(garch_train_vol.mean())
print("GARCH listo.")

# ── guardar ───────────────────────────────────────────────────────────────────
np.save(f"{SAVE_DIR}/y_te.npy",        y_te)
np.save(f"{SAVE_DIR}/bnn_mean.npy",    preds["mean"])
np.save(f"{SAVE_DIR}/bnn_epi.npy",     preds["epi_std"])
np.save(f"{SAVE_DIR}/bnn_alea.npy",    preds["alea_std"])
np.save(f"{SAVE_DIR}/bnn_total.npy",   preds["total_std"])
np.save(f"{SAVE_DIR}/garch_vol.npy",   garch_logvar)
np.save(f"{SAVE_DIR}/bnn_mean_tr.npy", preds_tr["mean"])
np.save(f"{SAVE_DIR}/bnn_epi_tr.npy",  preds_tr["epi_std"])

with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump({"y_mean": y_mean,
               "n_tr": n_tr, "n_te": n_te,
               "dates_te": dates_te,
               "ticker": CONFIG["ticker"],
               "start":  CONFIG["start"],
               "end":    CONFIG["end"],
               "garch_target_vol": garch_target_vol}, f)

print(f"\nListo. Resultados guardados en {OUT_DIR}/")
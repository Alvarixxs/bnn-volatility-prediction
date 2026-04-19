CONFIG = {
    "ticker": "^GSPC",

    # "ticker":  "NG=F", # gas
    # "ticker":  "CL=F", # petroleo
    # "ticker":  "EWZ", # brasil

    # "ticker":  "HG=F", # oro
    # "ticker":  "^N225", # NIKKEI

    "start":   "1900-01-01",
    "end":     "2026-04-01",

    "subperiodos": {
        "2008": ("2007-07-01", "2010-03-31"),
    },

    "window":    30,
    "epsilon":   1e-4,

    # Partición
    "train_frac": 0.70,

    # Arquitectura
    "hidden":     32,
    "prior_std":  1.0,
    "dropout":    0.2,

    # Entrenamiento
    "lr":         1e-3,
    "batch_size": 64,
    "max_epochs": 500,

    # Inferencia bayesiana
    "n_samples":  500,
\
}
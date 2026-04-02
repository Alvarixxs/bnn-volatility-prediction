CONFIG = {
    # "ticker":  "^GSPC", # sp500
    # "ticker":  "^VIX",
    "ticker":  "NG=F", # gas
    # "ticker":  "CL=F", # petroleo

    "start":   "1900-01-01",
    "end":     "2026-04-01",

    "subperiodos": {
        "Pre-crisis 2004-07": ("2004-01-01", "2007-06-30"),
        "Crisis 2008":        ("2007-07-01", "2009-12-31"),
        "Calma QE":           ("2010-01-01", "2017-12-31"),
        "Volmageddon 2018":   ("2018-01-01", "2019-12-31"),
        "COVID spike":        ("2020-01-01", "2021-12-31"),
        "Post-COVID":         ("2022-01-01", "2026-04-01"),
    },

    "window":    20,
    "epsilon":   1e-4,

    # Partición
    "train_frac": 0.70,
    "val_frac":   0.15,

    # Arquitectura
    "hidden":     32,
    "prior_std":  1.0,
    "dropout":    0.2,

    # Entrenamiento
    "lr":         1e-3,
    "batch_size": 64,
    "max_epochs": 400,
    "patience":   25,

    # Inferencia bayesiana
    "n_samples":  500,
\
}
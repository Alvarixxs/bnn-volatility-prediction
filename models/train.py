# train.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32)

def make_loaders(X_tr, y_tr, batch_size=64):
    ds = TensorDataset(to_t(X_tr), to_t(y_tr))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def elbo_loss(model, x, y, n_samples=3, n_train=1):
    obs_var = model.obs_std() ** 2
    nll = sum(
        (0.5*(model(x, sample=True) - y)**2 / obs_var
         + torch.log(model.obs_std())).mean()
        for _ in range(n_samples)
    ) / n_samples
    return nll + model.kl() / n_train, nll.item()

def train_det(model, X_tr, y_tr, X_val, y_val, cfg):
    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    dl  = make_loaders(X_tr, y_tr, cfg["batch_size"])
    best_val, best_state, no_imp = float("inf"), None, 0

    best_val   = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}  # ← aquí
    no_imp     = 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(to_t(X_val)), to_t(y_val)).item()

        if vl < best_val - 1e-4:
            best_val, best_state, no_imp = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model

def train_bnn(model, X_tr, y_tr, X_val, y_val, cfg):
    n_train = len(X_tr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    dl  = make_loaders(X_tr, y_tr, cfg["batch_size"])
    best_val, best_state, no_imp = float("inf"), None, 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss, _ = elbo_loss(model, xb, yb, n_train=n_train)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(to_t(X_val), sample=False),
                            to_t(y_val)).item()

        if vl < best_val - 1e-4:
            best_val, best_state, no_imp = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model

def predict_bnn(model, X_test, y_std, y_mean, n_samples=500):
    model.train()
    x = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = torch.stack([model(x, sample=True) for _ in range(n_samples)]).numpy()

    mean_n      = preds.mean(0)
    epi_std_n   = preds.std(0)
    alea_std_n  = model.obs_std().item()
    total_std_n = np.sqrt(epi_std_n**2 + alea_std_n**2)

    return {
        "mean":      mean_n * y_std + y_mean,
        "epi_std":   epi_std_n * y_std,
        "alea_std":  alea_std_n * y_std,
        "total_std": total_std_n * y_std,
    }
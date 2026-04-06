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

def train_bnn(model, X_tr, y_tr, cfg):
    n_train = len(X_tr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    dl  = make_loaders(X_tr, y_tr, cfg["batch_size"])

    for _ in range(cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss, _ = elbo_loss(model, xb, yb, n_train=n_train)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(dl)

    return model

def predict_bnn(model, X_test, y_std, y_mean, n_samples=500):
    model.train()
    x = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        raw_preds = torch.stack([model(x, sample=True)
                                 for _ in range(n_samples)])  # (S, N)

    raw_preds_orig = raw_preds * y_std + y_mean  # (S, N)
    sigma2 = (model.obs_std().item() * y_std) ** 2

    preds_r2 = torch.exp(raw_preds_orig + sigma2 / 2)  # (S, N)
    mean_r2 = preds_r2.mean(0).numpy()   # (N,)

    epi_var = preds_r2.var(0).numpy()    # varianza muestral (S-1 en denominador)
    epi_std = np.sqrt(epi_var)

    alea_var_per_sample = torch.exp(2 * raw_preds_orig + sigma2) * (np.exp(sigma2) - 1)
    alea_var = alea_var_per_sample.mean(0).numpy()
    alea_std = np.sqrt(alea_var)

    # varianza total
    total_std = np.sqrt(epi_var + alea_var)

    return {
        "mean":      mean_r2,       # E[r²|x,D] en escala original
        "epi_std":   epi_std,       # std epistémica en escala de r²
        "alea_std":  alea_std,      # std aleatórica en escala de r²
        "total_std": total_std,     # std total en escala de r²
    }
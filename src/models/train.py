import torch
from torch.utils.data import DataLoader, TensorDataset


def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32)

def make_loaders(X_tr, y_tr, batch_size=64):
    ds = TensorDataset(to_t(X_tr), to_t(y_tr))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def elbo_loss(model, x, y, n_samples=1, n_train=1):
    r2  = torch.exp(y)
    nll = 0.0
    for _ in range(n_samples):
        y_hat = model(x, sample=True)
        nll   = nll + (0.5*(y_hat - y) + 0.5*r2*torch.exp(-y_hat)).mean()
    nll = nll / n_samples
    return nll + model.kl() / n_train, nll.item()

def train_bnn(model, X_tr_n, y_tr, cfg):
    n_train = len(X_tr_n)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    dl  = make_loaders(X_tr_n, y_tr, cfg["batch_size"])

    history = []   # ← añadir

    for _ in range(cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss, _ = elbo_loss(
                model, xb, yb,
                n_samples=cfg.get("n_elbo_samples", 1),
                n_train=n_train,
            )
            loss.backward()
            opt.step()
            train_loss += loss.item()
        history.append(train_loss / len(dl))   # ← añadir

    return model, history   # ← modificar
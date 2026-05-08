import numpy as np
import torch


def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32)

def predict_bnn(model, X_n, n_samples):
    model.eval()
    X = to_t(X_n)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            y_hat = model(X, sample=True)         # log(ν²) en escala original
            samples.append(torch.exp(y_hat).numpy())

    samples  = np.array(samples)                  # (S, T)
    mean     = samples.mean(0)
    epi_var  = samples.var(0, ddof=1)
    alea_var = 2.0 * (samples**2).mean(0)         # Var[r²|x,θ] = 2ν⁴

    return {
        "mean":      mean,
        "epi_std":   np.sqrt(epi_var),
        "alea_std":  np.sqrt(alea_var),
        "total_std": np.sqrt(epi_var + alea_var),
    }
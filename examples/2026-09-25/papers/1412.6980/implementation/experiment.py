"""Sec 6.1 of Kingma & Ba: L2-regularised logistic regression, Adam vs Nesterov SGD vs Adagrad."""

import json

import numpy as np

from optim import Adagrad, Adam, NesterovSGD

rng = np.random.default_rng(0)
N, D, C, LAM, BATCH, EPOCHS = 6000, 784, 10, 1e-4, 128, 10
centers = rng.standard_normal((C, D)) * 0.06  # overlap tuned for MNIST-like training loss (~0.3)
y = rng.integers(0, C, N)
X = centers[y] + rng.standard_normal((N, D))
Y = np.eye(C)[y]


def loss_grad(W, xb, yb):
    z = xb @ W
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    loss = -np.mean(np.sum(yb * np.log(p + 1e-12), 1)) + LAM / 2 * np.sum(W * W)
    return loss, xb.T @ (p - yb) / len(xb) + LAM * W


def train(opt):
    W = np.zeros((D, C))
    order_rng = np.random.default_rng(1)  # same minibatch order for every optimizer
    for _ in range(EPOCHS):
        idx = order_rng.permutation(N)
        for s in range(0, N, BATCH):
            b = idx[s:s + BATCH]
            at = opt.lookahead(W) if hasattr(opt, "lookahead") else W
            _, g = loss_grad(at, X[b], Y[b])
            W = opt.step(W, g)
    final = loss_grad(W, X, Y)[0]
    return final if np.isfinite(final) else float("inf")


GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
best = {}
for name, make in [("adam", lambda a: Adam(a, decay=True)),
                   ("nesterov", lambda a: NesterovSGD(a, decay=True)),
                   ("adagrad", lambda a: Adagrad(a, decay=True))]:
    runs = {a: train(make(a)) for a in GRID}
    a_best = min(runs, key=runs.get)
    best[name] = (a_best, runs[a_best])

metrics = {
    **{f"{k}_final_train_loss": float(v[1]) for k, v in best.items()},
    **{f"{k}_best_alpha": v[0] for k, v in best.items()},
    "adam_faster_than_adagrad": bool(best["adam"][1] < best["adagrad"][1]),
    "adam_similar_to_nesterov": bool(abs(best["adam"][1] - best["nesterov"][1]) <= 0.1 * best["nesterov"][1]),
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))

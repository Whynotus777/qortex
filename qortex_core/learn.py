"""Training: behavior cloning and teacher -> student distillation, in numpy."""
import numpy as np
from .policy import Policy


def _init(sizes, rng):
    return [(rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2.0 / a),
             np.zeros(b, np.float32)) for a, b in zip(sizes[:-1], sizes[1:])]


def _fit(X, Y, hidden, epochs, lr, seed, log):
    rng = np.random.default_rng(seed)
    params = _init([X.shape[1], *hidden, Y.shape[1]], rng)
    flat = [p for wb in params for p in wb]
    m = [np.zeros_like(p) for p in flat]
    v = [np.zeros_like(p) for p in flat]
    n, bs, step, L = len(X), 256, 0, len(params)
    for ep in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            acts = [X[idx]]
            for i, (w, b) in enumerate(params):
                z = acts[-1] @ w + b
                acts.append(np.tanh(z) if i == L - 1 else np.maximum(z, 0))
            err = acts[-1] - Y[idx]
            total += float((err ** 2).sum())
            g = 2 * err / len(idx) * (1 - acts[-1] ** 2)
            grads = []
            for i in reversed(range(L)):
                w, b = params[i]
                grads.append((acts[i].T @ g, g.sum(0)))
                if i:
                    g = (g @ w.T) * (acts[i] > 0)
            grads = [p for wb in reversed(grads) for p in wb]
            step += 1
            cur = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
            for j, (p, gr) in enumerate(zip(flat, grads)):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr * gr
                p -= cur * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            log(f"epoch {ep + 1}/{epochs} mse={total / n / Y.shape[1]:.5f}")
    return params, total / n / Y.shape[1]


def behavior_clone(obs, actions, hidden=(256, 256), epochs=40, lr=1e-3, seed=0, meta=None, log=None):
    obs = np.asarray(obs, np.float32)
    actions = np.clip(np.asarray(actions, np.float32), -0.999, 0.999)
    mean, std = obs.mean(0), obs.std(0) + 1e-6
    params, loss = _fit((obs - mean) / std, actions, list(hidden), epochs, lr, seed, log)
    return Policy(params, mean, std, dict(meta or {}, train_mse=round(loss, 6)))


def distill(teacher, obs, hidden=(32, 32), epochs=60, lr=2e-3, seed=0, meta=None, log=None):
    """Train a smaller student to match the teacher's actions.

    The student sees the dataset observations plus jittered copies, all
    labelled by the teacher. No ground-truth actions are needed.
    """
    obs = np.asarray(obs, np.float32)
    rng = np.random.default_rng(seed)
    jitter = obs + 0.05 * teacher.obs_std * rng.standard_normal(obs.shape).astype(np.float32)
    X = np.concatenate([obs, jitter])
    Y = np.clip(teacher.act(X), -0.999, 0.999)
    params, loss = _fit((X - teacher.obs_mean) / teacher.obs_std, Y, list(hidden), epochs, lr, seed, log)
    return Policy(params, teacher.obs_mean, teacher.obs_std, dict(meta or {}, distill_mse=round(loss, 6)))

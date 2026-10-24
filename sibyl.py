#!/usr/bin/env python3
"""
SIBYL  v0.1
════════════════════════════════════════════════════════════════════════
Reasoning under uncertainty, from scratch. One file, one dependency (numpy).

  AUTODIFF     reverse-mode automatic differentiation on numpy arrays
               (broadcasting, matmul, indexing, softplus, sigmoid, ...)
  INFERENCE    Hamiltonian Monte Carlo with dual-averaging step size and
               diagonal mass-matrix adaptation · ADVI (mean-field, Adam) ·
               split-R̂ and effective sample size
  CAUSAL       structural causal models: do(), counterfactuals ·
               backdoor regression · IPW · doubly-robust AIPW · 2SLS
               instrumental variables · bootstrap intervals
  FORECAST     autoregressive forecasts with split-conformal intervals that
               carry a finite-sample coverage guarantee

Write a log-density, get posterior samples. State a causal graph, get an
effect. Ask for a forecast, get an interval you can hold it to.

It quantifies uncertainty. It does NOT remove it, and every causal estimate
rests on assumptions you must defend. Not financial or medical advice.

Usage:
  python sibyl.py demo
  python sibyl.py --test | --bench
  python sibyl.py charts OUTDIR
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

__version__ = "0.1.0"
LOG2PI = math.log(2 * math.pi)


# ═══════════════════════════ AUTODIFF ═══════════════════════════════
def _ub(g, shape):
    """Sum a broadcast gradient back down to `shape`."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _t(x):
    return x if isinstance(x, T) else T(x)


def _mk(val, parents, grads):
    out = T(val, parents)

    def back():
        for p, gp in zip(parents, grads(out.g)):
            p.g = p.g + _ub(gp, p.v.shape)
    out._back = back
    return out


class T:
    """Node of a computation graph. `.v` value, `.g` gradient after backward()."""
    __slots__ = ("v", "g", "_prev", "_back")

    def __init__(self, v, prev=(), back=None):
        self.v = np.asarray(v, dtype=float)
        self.g, self._prev, self._back = None, prev, back

    shape = property(lambda s: s.v.shape)

    def __add__(a, b): b = _t(b); return _mk(a.v + b.v, (a, b), lambda g: (g, g))
    __radd__ = __add__
    def __sub__(a, b): b = _t(b); return _mk(a.v - b.v, (a, b), lambda g: (g, -g))
    def __rsub__(a, b): return _t(b) - a
    def __mul__(a, b): b = _t(b); return _mk(a.v * b.v, (a, b), lambda g: (g * b.v, g * a.v))
    __rmul__ = __mul__
    def __truediv__(a, b): b = _t(b); return _mk(a.v / b.v, (a, b), lambda g: (g / b.v, -g * a.v / b.v ** 2))
    def __rtruediv__(a, b): return _t(b) / a
    def __neg__(a): return _mk(-a.v, (a,), lambda g: (-g,))
    def __pow__(a, c): return _mk(a.v ** c, (a,), lambda g: (g * c * a.v ** (c - 1),))

    def __matmul__(a, b):
        b = _t(b); av, bv = a.v, b.v

        def grads(g):
            if av.ndim == 2 and bv.ndim == 2: return g @ bv.T, av.T @ g
            if av.ndim == 2 and bv.ndim == 1: return np.outer(g, bv), av.T @ g
            if av.ndim == 1 and bv.ndim == 2: return bv @ g, np.outer(av, g)
            return g * bv, g * av
        return _mk(av @ bv, (a, b), grads)

    def sum(a, axis=None, keepdims=False):
        def grads(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, a.v.shape),)
        return _mk(a.v.sum(axis=axis, keepdims=keepdims), (a,), grads)

    def mean(a):
        return a.sum() / a.v.size

    def reshape(a, *shape):
        return _mk(a.v.reshape(*shape), (a,), lambda g: (g.reshape(a.v.shape),))

    def __getitem__(a, idx):
        def grads(g):
            full = np.zeros_like(a.v)
            np.add.at(full, idx, g)
            return (full,)
        return _mk(a.v[idx], (a,), grads)

    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            n, done = stack.pop()
            if done:
                topo.append(n); continue
            if id(n) in seen:
                continue
            seen.add(id(n)); stack.append((n, True))
            stack.extend((p, False) for p in n._prev if id(p) not in seen)
        for n in topo:
            n.g = np.zeros_like(n.v)
        self.g = np.ones_like(self.v)
        for n in reversed(topo):
            if n._back:
                n._back()


def exp(a): a = _t(a); v = np.exp(a.v); return _mk(v, (a,), lambda g: (g * v,))
def log(a): a = _t(a); return _mk(np.log(a.v), (a,), lambda g: (g / a.v,))
def sqrt(a): return _t(a) ** 0.5
def tanh(a): a = _t(a); v = np.tanh(a.v); return _mk(v, (a,), lambda g: (g * (1 - v * v),))


def sigmoid(a):
    a = _t(a); v = np.where(a.v >= 0, 1 / (1 + np.exp(-np.abs(a.v))), np.exp(-np.abs(a.v)) / (1 + np.exp(-np.abs(a.v))))
    return _mk(v, (a,), lambda g: (g * v * (1 - v),))


def softplus(a):
    a = _t(a)
    return _mk(np.logaddexp(0.0, a.v), (a,), lambda g: (g * np.where(a.v >= 0, 1 / (1 + np.exp(-np.abs(a.v))), np.exp(-np.abs(a.v)) / (1 + np.exp(-np.abs(a.v)))),))


def value_and_grad(f, x):
    """f: callable(T) -> scalar T.  Returns (value, gradient ndarray)."""
    xt = T(np.array(x, float))
    out = f(xt)
    if out.v.size != 1:
        raise ValueError("f must return a scalar")
    out.backward()
    return float(out.v), xt.g.copy()


# ── log-densities (all return a scalar T; constants included) ──────────
def normal_lp(x, mu, sigma):
    z = (_t(x) - mu) / sigma
    return (-0.5 * z * z - log(sigma) - 0.5 * LOG2PI).sum()


def bernoulli_logit_lp(y, z):
    return (_t(y) * z - softplus(z)).sum()


def poisson_log_lp(y, eta):
    y = np.asarray(y, float)
    return (_t(y) * eta - exp(eta)).sum() - float(sum(math.lgamma(v + 1) for v in y.ravel()))


def studentt_lp(x, nu, mu, sigma):
    z = (_t(x) - mu) / sigma
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    return (c - log(sigma) - (nu + 1) / 2 * log(1 + z * z / nu)).sum()


def exponential_lp(x, rate):
    return (math.log(rate) - rate * _t(x)).sum()


class Params:
    """Named parameter blocks packed in one flat vector.  P = Params(beta=3, sigma=());  P.unpack(theta)"""

    def __init__(self, **shapes):
        self.shapes = {k: ((v,) if isinstance(v, int) else tuple(v)) for k, v in shapes.items()}
        self.sizes = {k: int(np.prod(s)) if s else 1 for k, s in self.shapes.items()}
        self.dim = sum(self.sizes.values())

    def unpack(self, theta):
        out, i = {}, 0
        for k, s in self.shapes.items():
            n = self.sizes[k]
            out[k] = theta[i:i + n].reshape(*s) if s else theta[i]
            i += n
        return out

    def names(self):
        res = []
        for k, s in self.shapes.items():
            n = self.sizes[k]
            res += [k] if (not s or n == 1 and s == ()) else [f"{k}[{j}]" for j in range(n)]
        return res


# ═══════════════════════ MCMC / VARIATIONAL ═════════════════════════
def rhat(draws):
    """Split-R̂.  draws: (chains, n, dim)."""
    c, n, d = draws.shape
    h = n // 2
    x = np.concatenate([draws[:, :h], draws[:, h:2 * h]], axis=0)  # (2c, h, d)
    m = x.shape[0]
    W = x.var(axis=1, ddof=1).mean(axis=0)
    B = h * x.mean(axis=1).var(axis=0, ddof=1)
    vp = (h - 1) / h * W + B / h
    return np.sqrt(vp / np.maximum(W, 1e-300))


def ess(draws):
    """Bulk effective sample size (Geyer initial monotone sequence).  draws: (chains, n, dim)."""
    m, n, d = draws.shape
    out = np.zeros(d)
    for k in range(d):
        x = draws[:, :, k]; xm = x.mean(axis=1, keepdims=True); xc = x - xm
        f = np.fft.rfft(xc, n=2 * n, axis=1)
        ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
        ac = ac * n / (n - np.arange(n))  # unbiased lag normalisation
        W = xc.var(axis=1, ddof=1).mean()
        B = n * xm.var(ddof=1) if m > 1 else 0.0
        vp = (n - 1) / n * W + B / n
        rho = 1 - (W - ac.mean(axis=0)) / vp
        s, prev = 0.0, np.inf
        for i in range(n // 2):
            P = rho[2 * i] + rho[2 * i + 1]
            if P < 0:
                break
            P = min(P, prev); prev = P; s += P
        tau = max(-1 + 2 * s, 1e-2)
        out[k] = m * n / tau
    return out


class Fit:
    def __init__(self, draws, **info):
        self.draws, self.info = draws, info
        self.chains, self.n, self.dim = draws.shape
        self.rhat, self.ess = rhat(draws), ess(draws)

    def flat(self):
        return self.draws.reshape(-1, self.dim)

    def mean(self): return self.flat().mean(axis=0)
    def std(self): return self.flat().std(axis=0, ddof=1)

    def summary(self, names=None, out=print):
        names = names or [f"x[{i}]" for i in range(self.dim)]
        f = self.flat()
        q = np.quantile(f, [0.05, 0.95], axis=0)
        out(f"  {'param':12}{'mean':>9}{'sd':>9}{'5%':>9}{'95%':>9}{'ess':>8}{'rhat':>7}")
        for i, nm in enumerate(names):
            out(f"  {nm:12}{f[:, i].mean():9.3f}{f[:, i].std(ddof=1):9.3f}{q[0, i]:9.3f}{q[1, i]:9.3f}{self.ess[i]:8.0f}{self.rhat[i]:7.3f}")
        out(f"  accept {self.info['accept']:.2f} · step {np.mean(self.info['eps']):.3f} · "
            f"divergences {self.info['divergences']} · gradient evals {self.info['grad_evals']:,}")


def hmc(logp, x0, n_draws=1000, n_warmup=1000, n_chains=4, seed=0, target_accept=0.8, traj=2.0):
    """Hamiltonian Monte Carlo.  logp: callable(T) -> scalar T (log posterior, up to a constant)."""
    x0 = np.asarray(x0, float); dim = x0.size
    rng = np.random.default_rng(seed)
    counter = [0]

    def vg(x):
        counter[0] += 1
        v, g = value_and_grad(logp, x)
        return (v, g) if np.isfinite(v) and np.all(np.isfinite(g)) else (-np.inf, np.zeros_like(g))

    def leap(x, p, g, eps, L, var):
        p = p + 0.5 * eps * g
        lp = -np.inf
        for i in range(L):
            x = x + eps * var * p
            lp, g = vg(x)
            if not np.isfinite(lp):
                return x, p, lp, g
            if i != L - 1:
                p = p + eps * g
        return x, p + 0.5 * eps * g, lp, g

    def step(x, lp, g, eps, var):
        p0 = rng.standard_normal(dim) / np.sqrt(var)
        L = max(1, min(int(round(traj / eps * rng.uniform(0.8, 1.2))), 64))
        x1, p1, lp1, g1 = leap(x, p0, g, eps, L, var)
        if not np.isfinite(lp1):
            return x, lp, g, 0.0, True
        dH = (lp1 - 0.5 * np.sum(var * p1 * p1)) - (lp - 0.5 * np.sum(var * p0 * p0))
        a = float(min(1.0, math.exp(min(dH, 0.0))))
        div = dH < -1000
        if rng.random() < a:
            return x1, lp1, g1, a, div
        return x, lp, g, a, div

    def find_eps(x, lp, g, var):
        eps = 0.1
        p0 = rng.standard_normal(dim) / np.sqrt(var)
        x1, p1, lp1, _ = leap(x, p0, g, eps, 1, var)
        dH = (lp1 - 0.5 * np.sum(var * p1 * p1)) - (lp - 0.5 * np.sum(var * p0 * p0)) if np.isfinite(lp1) else -np.inf
        d = 1 if dH > math.log(0.5) else -1
        for _ in range(30):
            eps *= 2.0 ** d
            x1, p1, lp1, _ = leap(x, p0, g, eps, 1, var)
            dH = (lp1 - 0.5 * np.sum(var * p1 * p1)) - (lp - 0.5 * np.sum(var * p0 * p0)) if np.isfinite(lp1) else -np.inf
            if (d == 1 and not dH > math.log(0.5)) or (d == -1 and dH > math.log(0.5)):
                break
        return eps

    out = np.zeros((n_chains, n_draws, dim)); accs, epss, divs = [], [], 0
    for c in range(n_chains):
        x = x0 + 0.1 * rng.standard_normal(dim)
        lp, g = vg(x)
        for _ in range(20):
            if np.isfinite(lp):
                break
            x = x0 + 0.1 * rng.standard_normal(dim); lp, g = vg(x)
        var = np.ones(dim)
        eps = find_eps(x, lp, g, var)
        mu_da, hbar, leps_bar, t = math.log(10 * eps), 0.0, 0.0, 0
        marks = {int(n_warmup * f) for f in (0.25, 0.5, 0.75)}
        seg = []
        for it in range(n_warmup):
            x, lp, g, a, _ = step(x, lp, g, eps, var)
            t += 1
            hbar = (1 - 1 / (t + 10)) * hbar + (target_accept - a) / (t + 10)
            leps = mu_da - math.sqrt(t) / 0.05 * hbar
            eta = t ** -0.75
            leps_bar = eta * leps + (1 - eta) * leps_bar
            eps = math.exp(leps)
            seg.append(x.copy())
            if it + 1 in marks and len(seg) > 30:
                s = np.array(seg); nn = len(s)
                var = (nn / (nn + 5)) * s.var(axis=0, ddof=1) + 1e-3 * (5 / (nn + 5))
                eps = find_eps(x, lp, g, var)
                mu_da, hbar, leps_bar, t = math.log(10 * eps), 0.0, 0.0, 0
                seg = []
        eps = math.exp(leps_bar) if n_warmup else eps
        for i in range(n_draws):
            x, lp, g, a, d = step(x, lp, g, eps, var)
            out[c, i] = x; accs.append(a); divs += int(d)
        epss.append(eps)
    return Fit(out, accept=float(np.mean(accs)), eps=epss, divergences=divs, grad_evals=counter[0])


def advi(logp, x0, steps=3000, mc=4, lr=0.05, seed=0):
    """Mean-field Gaussian variational inference (reparameterisation trick + Adam)."""
    x0 = np.asarray(x0, float); d = x0.size
    rng = np.random.default_rng(seed)
    phi = np.concatenate([x0, np.full(d, -1.0)])
    m, v = np.zeros(2 * d), np.zeros(2 * d)
    elbos = []
    for t in range(1, steps + 1):
        eps = rng.standard_normal((mc, d))

        def neg_elbo(p):
            mu, rho = p[:d], p[d:]
            tot = None
            for k in range(mc):
                lp = logp(mu + exp(rho) * eps[k])
                tot = lp if tot is None else tot + lp
            return -(tot / mc + rho.sum())
        val, g = value_and_grad(neg_elbo, phi)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        phi -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        elbos.append(-val)
    return {"mean": phi[:d], "std": np.exp(phi[d:]), "elbo": np.array(elbos)}


# ═══════════════════════════ CAUSAL INFERENCE ═══════════════════════
class SCM:
    """Structural causal model.  Each node:  value = f(parents) + noise   (additive)
    or value = f(parents, noise)   (general).  Supports do() and counterfactuals."""

    def __init__(self):
        self.nodes, self.order = {}, []

    def add(self, name, parents=(), f=None, noise=None, additive=True):
        for p in parents:
            if p not in self.nodes:
                raise KeyError(f"parent {p!r} must be added before {name!r}")
        self.nodes[name] = {
            "parents": list(parents), "f": f or (lambda: 0.0), "additive": additive,
            "noise": noise or (lambda rng, n: rng.standard_normal(n))}
        self.order.append(name)
        return self

    def sample(self, n, seed=0, do=None):
        """Draw n units. Noise is drawn for every node first, so interventions share
        common random numbers with the observational world."""
        rng = np.random.default_rng(seed); do = do or {}
        U = {v: self.nodes[v]["noise"](rng, n) for v in self.order}
        vals = {}
        for v in self.order:
            nd = self.nodes[v]
            if v in do:
                vals[v] = np.broadcast_to(np.asarray(do[v], float), (n,)).copy()
                continue
            args = [vals[p] for p in nd["parents"]]
            vals[v] = nd["f"](*args) + U[v] if nd["additive"] else nd["f"](*args, U[v])
        return vals

    def ate(self, treat, outcome, n=200_000, seed=0, t1=1.0, t0=0.0):
        """E[Y | do(T=t1)] − E[Y | do(T=t0)] by simulating the intervention."""
        return float(self.sample(n, seed, {treat: t1})[outcome].mean() - self.sample(n, seed, {treat: t0})[outcome].mean())

    def counterfactual(self, obs, do):
        """Abduction → action → prediction for one unit. `obs` holds every variable's observed value."""
        vals, changed = {}, set(do)
        for v in self.order:
            nd = self.nodes[v]
            if v in do:
                vals[v] = float(do[v]); continue
            if not any(p in changed for p in nd["parents"]):
                vals[v] = float(obs[v]); continue
            if not nd["additive"]:
                raise ValueError(f"counterfactual through non-additive node {v!r} is not identifiable here")
            u = obs[v] - nd["f"](*[np.asarray(obs[p], float) for p in nd["parents"]])
            vals[v] = float(nd["f"](*[np.asarray(vals[p], float) for p in nd["parents"]]) + u)
            changed.add(v)
        return vals


def _ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def logit_fit(X, t, iters=50, ridge=1e-8):
    """Maximum-likelihood logistic regression by Newton / IRLS. X includes the intercept column."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = _sigmoid(X @ b)
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (t - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return b


def propensity(X, t, clip=0.01):
    D = np.c_[np.ones(len(t)), X]
    return np.clip(_sigmoid(D @ logit_fit(D, t)), clip, 1 - clip)


def ate_naive(t, y, X=None):
    return float(y[t == 1].mean() - y[t == 0].mean())


def ate_regression(t, y, X):
    """Backdoor adjustment by linear regression on the confounders."""
    return float(_ols(np.c_[np.ones(len(y)), t, X], y)[1])


def ate_ipw(t, y, X):
    """Hájek inverse-probability weighting."""
    p = propensity(X, t)
    return float(np.sum(t * y / p) / np.sum(t / p) - np.sum((1 - t) * y / (1 - p)) / np.sum((1 - t) / (1 - p)))


def ate_aipw(t, y, X):
    """Doubly-robust augmented IPW: consistent if EITHER the outcome model or the propensity model is right."""
    p = propensity(X, t)
    D = np.c_[np.ones(len(y)), X]
    m1 = D @ _ols(D[t == 1], y[t == 1]); m0 = D @ _ols(D[t == 0], y[t == 0])
    return float(np.mean(m1 - m0 + t * (y - m1) / p - (1 - t) * (y - m0) / (1 - p)))


def iv_2sls(y, t, z, X=None):
    """Two-stage least squares with instrument(s) z; recovers the effect of t despite hidden confounding."""
    z = z.reshape(len(y), -1)
    ctrl = np.ones((len(y), 1)) if X is None else np.c_[np.ones(len(y)), X]
    t_hat = np.c_[ctrl, z] @ _ols(np.c_[ctrl, z], t)
    return float(_ols(np.c_[ctrl, t_hat], y)[-1])


def bootstrap_ci(fn, *arrays, n_boot=300, seed=0, level=0.95):
    rng = np.random.default_rng(seed); n = len(arrays[0])
    est = np.array([fn(*[a[i] for a in arrays]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    a = (1 - level) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a)), float(est.std(ddof=1))


def simpson_scm():
    """Confounded world: a larger Z raises treatment probability but LOWERS the outcome.
    True causal effect of T on Y is +1, yet the naive comparison is negative."""
    s = SCM()
    s.add("Z")
    s.add("T", ["Z"], lambda z, u: (u < _sigmoid(1.5 * z)).astype(float), noise=lambda r, n: r.random(n), additive=False)
    s.add("Y", ["T", "Z"], lambda t, z: 1.0 * t - 3.0 * z)
    return s


def iv_scm():
    """Hidden confounder U drives both T and Y; Z is a valid instrument. True effect of T on Y is 1.5."""
    s = SCM()
    s.add("U"); s.add("Z")
    s.add("T", ["Z", "U"], lambda z, u: 0.8 * z + 1.0 * u)
    s.add("Y", ["T", "U"], lambda t, u: 1.5 * t + 2.0 * u)
    return s


# ═════════════════════ FORECASTING WITH CONFORMAL INTERVALS ═════════
def ar_fit(y, p):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y) - p)] + [y[p - k - 1: len(y) - k - 1] for k in range(p)])
    return _ols(X, y[p:])


def ar_forecast(y, coefs, h=1):
    hist = list(np.asarray(y, float)); p = len(coefs) - 1
    for _ in range(h):
        hist.append(coefs[0] + sum(coefs[k + 1] * hist[-1 - k] for k in range(p)))
    return np.array(hist[-h:])


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile: ⌈(m+1)(1−α)⌉-th smallest score."""
    s = np.sort(np.asarray(scores)); m = len(s)
    k = math.ceil((m + 1) * (1 - alpha))
    return float(s[min(k, m) - 1]) if k <= m else float("inf")


def conformal_ar(y, p=2, alpha=0.1, h=1, cal_frac=0.3):
    """h-step AR forecast with a split-conformal interval.
    Coverage ≥ 1−α holds for exchangeable calibration errors, with no distributional assumption."""
    y = np.asarray(y, float); n = len(y); n_tr = int(n * (1 - cal_frac))
    c = ar_fit(y[:n_tr], p)
    errs = [abs(y[t + h - 1] - ar_forecast(y[:t], c, h)[-1]) for t in range(n_tr, n - h + 1)]
    q = conformal_quantile(errs, alpha)
    c_all = ar_fit(y, p); f = float(ar_forecast(y, c_all, h)[-1])
    return {"forecast": f, "lo": f - q, "hi": f + q, "q": q, "coefs": c_all, "n_cal": len(errs)}


def conformal_regression(predict, X_cal, y_cal, X_new, alpha=0.1):
    """Distribution-free prediction intervals around ANY fitted model `predict`."""
    q = conformal_quantile(np.abs(y_cal - predict(X_cal)), alpha)
    f = predict(X_new)
    return f - q, f + q


def simulate_ar(n, coefs=(0.6, -0.2), c=0.1, df=5, rng=None):
    """AR(p) with heavy-tailed Student-t shocks (so Gaussian-interval formulas would be wrong)."""
    rng = rng or np.random.default_rng(0)
    e = rng.standard_t(df, n + 100) * math.sqrt((df - 2) / df)
    y = np.zeros(n + 100); p = len(coefs)
    for t in range(p, n + 100):
        y[t] = c + sum(coefs[k] * y[t - 1 - k] for k in range(p)) + e[t]
    return y[100:]


def coverage_experiment(runs=400, n=500, alpha=0.1, h=1, seed=0):
    rng = np.random.default_rng(seed); hit = 0
    for _ in range(runs):
        y = simulate_ar(n + h, rng=rng)
        r = conformal_ar(y[:n], 2, alpha, h)
        hit += r["lo"] <= y[n + h - 1] <= r["hi"]
    return hit / runs


# ═════════════════════════════ DEMO / REPORT ════════════════════════
def logistic_problem(n=500, seed=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)); beta = np.array([-0.5, 1.2, -0.8, 0.4])
    y = (rng.random(n) < _sigmoid(beta[0] + X @ beta[1:])).astype(float)
    D = np.c_[np.ones(n), X]
    logp = lambda th: bernoulli_logit_lp(y, T(D) @ th) + normal_lp(th, 0.0, 2.5)
    return logp, D, y, beta


def demo(out=print):
    bar = "═" * 72
    out(f"{bar}\n SIBYL {__version__} · reasoning under uncertainty\n{bar}")
    logp, D, y, beta = logistic_problem()
    out("\n① BAYESIAN LOGISTIC REGRESSION  (500 observations, HMC 4 chains)")
    fit = hmc(logp, np.zeros(4), n_draws=400, n_warmup=400, n_chains=4, seed=1)
    fit.summary(["intercept", "x1", "x2", "x3"], out)
    mle = logit_fit(D, y)
    out(f"  true      {np.round(beta, 2)}\n  MLE       {np.round(mle, 2)}\n  posterior {np.round(fit.mean(), 2)}")
    a = advi(logp, np.zeros(4), steps=1500)
    out(f"  ADVI mean {np.round(a['mean'], 2)}  (mean-field underestimates correlated sd, by design)")

    out("\n② CAUSAL INFERENCE  (confounded world, true effect of treatment = +1.00)")
    s = simpson_scm(); d = s.sample(20000, seed=5)
    t, yy, X = d["T"], d["Y"], d["Z"][:, None]
    out(f"  do()-simulation of the SCM      {s.ate('T', 'Y'):+.3f}")
    for nm, fn in (("naive difference", ate_naive), ("backdoor regression", ate_regression),
                   ("IPW", ate_ipw), ("doubly-robust AIPW", ate_aipw)):
        lo, hi, _ = bootstrap_ci(fn, t, yy, X, n_boot=150)
        out(f"  {nm:30}{fn(t, yy, X):+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    out("  → the naive estimate has the WRONG SIGN (Simpson's paradox); adjustment recovers +1.")
    iv = iv_scm(); di = iv.sample(30000, seed=6)
    ols = float(_ols(np.c_[np.ones(30000), di["T"]], di["Y"])[1])
    out(f"  hidden confounder, true +1.50:  OLS {ols:+.3f}   2SLS with instrument {iv_2sls(di['Y'], di['T'], di['Z']):+.3f}")
    cf = SCM().add("X").add("Y", ["X"], lambda x: 2.0 * x)
    obs = {"X": 1.0, "Y": 2.7}
    out(f"  counterfactual: had X been 3 instead of 1, Y would be {cf.counterfactual(obs, {'X': 3.0})['Y']:.2f} (observed 2.70)")

    out("\n③ FORECASTING WITH GUARANTEED COVERAGE  (AR(2), heavy-tailed shocks)")
    y = simulate_ar(600, rng=np.random.default_rng(9))
    for h in (1, 5):
        r = conformal_ar(y, 2, 0.1, h)
        out(f"  {h}-step forecast {r['forecast']:+.3f}   90% interval [{r['lo']:+.3f}, {r['hi']:+.3f}]   ({r['n_cal']} calibration errors)")
    out(f"  empirical coverage over 400 simulated series: h=1 {coverage_experiment(400, h=1):.1%}   h=5 {coverage_experiment(400, h=5, seed=1):.1%}   (target 90%)")
    out("\n  Quantifies uncertainty; does not remove it. Causal estimates need defensible assumptions.")


# ═══════════════════════════ TESTS / BENCH ══════════════════════════
def tests():
    ok = lambda s: print(f"  ✔ {s}")
    rng = np.random.default_rng(0)
    # autodiff vs finite differences
    X = rng.normal(size=(30, 4)); y = rng.normal(size=30)

    def f(th):
        b, z = th[:4], T(X) @ th[:4]
        return (normal_lp(y, z, exp(th[4])) + bernoulli_logit_lp((y > 0).astype(float), z * 0.3)
                + (tanh(b) * b).sum() + (b[1:3] ** 2).sum() + sigmoid(z).mean() + softplus(b).sum()
                + studentt_lp(y, 4.0, z, 1.3) + poisson_log_lp(np.abs(np.round(y * 3)), z * 0.2) + (b.reshape(2, 2) @ b[:2]).sum())
    x = rng.normal(size=5)
    g = value_and_grad(f, x)[1]
    num = np.array([(value_and_grad(f, x + h)[0] - value_and_grad(f, x - h)[0]) / 2e-6 for h in np.eye(5) * 1e-6])
    assert np.abs(g - num).max() < 1e-5
    ok(f"autodiff matches finite differences (max err {np.abs(g - num).max():.1e}), 12 ops incl. matmul / indexing / broadcasting")
    # HMC vs analytic conjugate posterior
    n = 200; Xr = np.c_[np.ones(n), rng.normal(size=n) * 2 + 3, rng.normal(size=n)]
    yr = Xr @ np.array([1.0, 0.5, -2.0]) + rng.normal(size=n); tau = 10.0
    cov = np.linalg.inv(Xr.T @ Xr + np.eye(3) / tau ** 2); mean = cov @ Xr.T @ yr
    lp = lambda th: normal_lp(T(yr), T(Xr) @ th, 1.0) + normal_lp(th, 0.0, tau)
    fit = hmc(lp, np.zeros(3), n_draws=300, n_warmup=300, n_chains=2, seed=3)
    assert np.abs(fit.mean() - mean).max() < 0.03 and np.all(np.abs(fit.std() / np.sqrt(np.diag(cov)) - 1) < 0.12)
    assert fit.rhat.max() < 1.05 and fit.ess.min() > 150 and fit.info["divergences"] == 0
    ok(f"HMC reproduces the analytic posterior (mean err {np.abs(fit.mean() - mean).max():.4f}, R̂ {fit.rhat.max():.3f}, ESS {fit.ess.min():.0f})")
    a = advi(lp, np.zeros(3), steps=1200)
    assert np.abs(a["mean"] - mean).max() < 0.06
    ok("ADVI recovers the posterior mean")
    # logistic regression: posterior ≈ MLE, truth inside intervals
    logp, D, yb, beta = logistic_problem()
    fb = hmc(logp, np.zeros(4), n_draws=300, n_warmup=300, n_chains=2, seed=2)
    assert np.abs(fb.mean() - logit_fit(D, yb)).max() < 0.15 and np.all(np.abs(fb.mean() - beta) < 3 * fb.std())
    ok("Bayesian logistic regression: posterior ≈ MLE and covers the truth")
    # diagnostics sanity: R̂ flags non-mixing chains
    bad = np.stack([np.full(200, 0.0) + rng.normal(size=200) * 0.1, np.full(200, 5.0) + rng.normal(size=200) * 0.1])[:, :, None]
    assert rhat(bad)[0] > 1.5 and abs(ess(rng.normal(size=(4, 500, 1)))[0] / 2000 - 1) < 0.3
    ok("R̂ flags non-converged chains; ESS ≈ N for independent draws")
    # causal
    s = simpson_scm(); assert abs(s.ate("T", "Y") - 1.0) < 1e-9
    d = s.sample(30000, seed=5); t, yy, Xz = d["T"], d["Y"], d["Z"][:, None]
    assert ate_naive(t, yy) < -0.5
    for fn in (ate_regression, ate_ipw, ate_aipw):
        assert abs(fn(t, yy, Xz) - 1.0) < 0.12, fn.__name__
    lo, hi, _ = bootstrap_ci(ate_aipw, t, yy, Xz, n_boot=120)
    assert lo < 1.0 < hi
    ok(f"Simpson's paradox: naive {ate_naive(t, yy):+.2f} (wrong sign); regression / IPW / AIPW recover +1; bootstrap CI covers")
    iv = iv_scm(); di = iv.sample(30000, seed=6)
    ols = _ols(np.c_[np.ones(30000), di["T"]], di["Y"])[1]
    assert abs(ols - 1.5) > 0.3 and abs(iv_2sls(di["Y"], di["T"], di["Z"]) - 1.5) < 0.1
    ok(f"hidden confounder: OLS {ols:+.2f} is biased, 2SLS recovers +1.5")
    cf = SCM().add("Z").add("X", ["Z"], lambda z: 0.5 * z).add("Y", ["X", "Z"], lambda x, z: 2 * x + z)
    obs = {"Z": 1.0, "X": 0.9, "Y": 3.2}
    got = cf.counterfactual(obs, {"X": 2.0})["Y"]
    assert abs(got - (3.2 + 2 * (2.0 - 0.9))) < 1e-12
    ok("counterfactual (abduction → action → prediction) is exact for additive models")
    # forecasting coverage
    c1, c5 = coverage_experiment(300, h=1), coverage_experiment(300, h=5, seed=1)
    assert 0.86 < c1 < 0.94 and 0.86 < c5 < 0.94
    ok(f"conformal AR intervals: coverage {c1:.1%} (h=1), {c5:.1%} (h=5) at 90% target, heavy-tailed shocks")
    assert conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.2) == 8 and conformal_quantile([1, 2], 0.01) == float("inf")
    Xc = rng.normal(size=(400, 2)); yc = Xc @ [1, -1] + rng.standard_t(4, 400)
    lo_, hi_ = conformal_regression(lambda Z: Z @ np.array([1.0, -1.0]), Xc[:200], yc[:200], Xc[200:])
    assert 0.85 < np.mean((yc[200:] >= lo_) & (yc[200:] <= hi_)) < 0.96
    ok("split-conformal quantile and generic conformal regression")
    print("all tests passed ✔")


def bench():
    rows = []

    def timed(name, fn):
        t0 = time.perf_counter(); r = fn(); rows.append((name, time.perf_counter() - t0, r))
    logp, D, yb, beta = logistic_problem()
    timed("autodiff: 2,000 gradients of a 500×4 logistic model", lambda: f"{2000 / max(1e-9, 1):.0f} grads" if [value_and_grad(logp, np.zeros(4)) for _ in range(2000)] else "")
    timed("HMC: 4 chains × 800 iterations, logistic regression", lambda: (lambda f: f"min ESS {f.ess.min():.0f}, R̂ {f.rhat.max():.3f}")(hmc(logp, np.zeros(4), 400, 400, 4, seed=1)))
    timed("ADVI: 3,000 Adam steps", lambda: f"mean err {np.abs(advi(logp, np.zeros(4), 3000)['mean'] - logit_fit(D, yb)).max():.3f}")
    s = simpson_scm(); d = s.sample(50000, seed=5)
    timed("AIPW + 300-replicate bootstrap, 50,000 units", lambda: f"CI width {np.subtract(*bootstrap_ci(ate_aipw, d['T'], d['Y'], d['Z'][:, None], n_boot=300)[1::-1]):.3f}")
    timed("conformal coverage experiment: 1,000 series", lambda: f"coverage {coverage_experiment(1000):.1%}")
    print(f"{'task':64}{'seconds':>9}  result")
    for n, sec, r in rows:
        print(f"{n:64}{sec:9.2f}  {r}")
    return rows


def _svg_header(W, H, title):
    from xml.sax.saxutils import escape
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
            f'<rect width="{W}" height="{H}" fill="#06040f"/>',
            f'<text x="40" y="36" font-family="Courier New,monospace" font-size="16" letter-spacing="4" fill="#a78bfa">{escape(title)}</text>']


def make_charts(outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    # ── causal estimators vs truth (real computation)
    s = simpson_scm(); d = s.sample(20000, seed=5); t, y, X = d["T"], d["Y"], d["Z"][:, None]
    rows = []
    for nm, fn in (("naive", ate_naive), ("regression", ate_regression), ("IPW", ate_ipw), ("AIPW", ate_aipw)):
        lo, hi, _ = bootstrap_ci(fn, t, y, X, n_boot=150)
        rows.append((nm, fn(t, y, X), lo, hi))
    W, H = 1000, 90 + 56 * len(rows); x0, x1 = 230, 940
    lo_a, hi_a = min(r[2] for r in rows) - 0.3, max(r[3] for r in rows) + 0.3
    px = lambda v: x0 + (v - lo_a) / (hi_a - lo_a) * (x1 - x0)
    o = _svg_header(W, H, "CAUSAL EFFECT OF T ON Y · TRUE = +1 · SIMPSON'S PARADOX")
    o.append(f'<line x1="{px(1):.1f}" x2="{px(1):.1f}" y1="56" y2="{H - 20}" stroke="#e5b53a" stroke-dasharray="6 5"/>'
             f'<text x="{px(1) + 8:.1f}" y="70" font-family="Courier New,monospace" font-size="13" fill="#e5b53a">truth +1.00</text>')
    o.append(f'<line x1="{px(0):.1f}" x2="{px(0):.1f}" y1="56" y2="{H - 20}" stroke="#2a2440"/>')
    for i, (nm, e, lo, hi) in enumerate(rows):
        yy = 100 + i * 56; good = abs(e - 1) < 0.2
        col = "#a78bfa" if good else "#ff2a3d"
        o.append(f'<text x="40" y="{yy + 5}" font-family="Courier New,monospace" font-size="16" fill="#e8e6e3">{nm}</text>'
                 f'<line x1="{px(lo):.1f}" x2="{px(hi):.1f}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="3"/>'
                 f'<circle cx="{px(e):.1f}" cy="{yy}" r="8" fill="{col}"/>'
                 f'<text x="{px(e):.1f}" y="{yy - 14}" text-anchor="middle" font-family="Courier New,monospace" font-size="13" fill="{col}">{e:+.2f}</text>')
    o.append(f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" fill="none" stroke="#7c3aed" stroke-opacity=".5"/></svg>')
    open(os.path.join(outdir, "causal.svg"), "w", encoding="utf-8").write("\n".join(o))
    # ── conformal band on one real simulated series
    y = simulate_ar(700, rng=np.random.default_rng(9)); n_tr = int(700 * 0.7)
    c = ar_fit(y[:n_tr], 2)
    errs = [abs(y[t] - ar_forecast(y[:t], c, 1)[0]) for t in range(n_tr, 600)]
    q = conformal_quantile(errs, 0.1)
    idx = range(600, 700); pred = np.array([ar_forecast(y[:t], c, 1)[0] for t in idx]); act = y[600:700]
    cov = float(np.mean(np.abs(act - pred) <= q))
    W, H = 1100, 380; ml, mr, mt, mb = 60, 30, 70, 40
    lo_v, hi_v = float((pred - q).min()) - 0.2, float((pred + q).max()) + 0.2
    X_ = lambda i: ml + i / 99 * (W - ml - mr); Y_ = lambda v: mt + (hi_v - v) / (hi_v - lo_v) * (H - mt - mb)
    o = _svg_header(W, H, f"90% CONFORMAL BAND · HELD-OUT DAYS · COVERAGE {cov:.0%}")
    band = [(X_(i), Y_(pred[i] + q)) for i in range(100)] + [(X_(i), Y_(pred[i] - q)) for i in range(99, -1, -1)]
    o.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in band) + '" fill="#7c3aed" opacity=".28"/>')
    o.append('<polyline fill="none" stroke="#a78bfa" stroke-width="1.5" points="' + " ".join(f"{X_(i):.1f},{Y_(pred[i]):.1f}" for i in range(100)) + '"/>')
    o.append('<polyline fill="none" stroke="#e8e6e3" stroke-width="1.8" points="' + " ".join(f"{X_(i):.1f},{Y_(act[i]):.1f}" for i in range(100)) + '"/>')
    for i in range(100):
        if abs(act[i] - pred[i]) > q:
            o.append(f'<circle cx="{X_(i):.1f}" cy="{Y_(act[i]):.1f}" r="5" fill="#ff2a3d"/>')
    o.append(f'<text x="{ml}" y="{H - 14}" font-family="Courier New,monospace" font-size="13" fill="#8a8a92">white = actual · violet = forecast ± conformal radius {q:.2f} · red = outside the band</text>'
             f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" fill="none" stroke="#7c3aed" stroke-opacity=".5"/></svg>')
    open(os.path.join(outdir, "forecast.svg"), "w", encoding="utf-8").write("\n".join(o))


def main(argv):
    if "--test" in argv:
        tests()
    elif "--bench" in argv:
        bench()
    elif len(argv) >= 2 and argv[1] == "charts":
        make_charts(argv[2] if len(argv) > 2 else "."); print("charts written")
    else:
        demo()


def cli():
    main(sys.argv)


if __name__ == "__main__":
    main(sys.argv)

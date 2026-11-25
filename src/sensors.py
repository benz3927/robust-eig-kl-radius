"""A design problem where the best design depends on how much you trust the prior.

theta ~ N(0, 1)  (reference prior p).
Design = a "sensor" k with centre c_k and width w_k:
    y | theta, k ~ N( tanh((theta - c_k) / w_k),  s^2 )
A narrow sensor (small w) is very sharp near its centre but saturates (gives
almost no information) once theta is far from c_k. A wide sensor is less sharp
but informative over a broad range. Think "zoom lens vs wide-angle lens".

All quantities are computed on a fine grid (theta is 1-D), so they are accurate
to quadrature error, with no Monte Carlo noise. Sampling estimators are separate.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

S_NOISE = 0.05
SENSORS = {  # name: (centre, width)
    "sharp": (0.0, 0.5),
    "broad": (0.0, 2.5),
}
TH = np.linspace(-8, 8, 2001)
Y = np.linspace(-1 - 6 * S_NOISE, 1 + 6 * S_NOISE, 3201)
DY = Y[1] - Y[0]


def mean_fn(theta, sensor):
    c, w = SENSORS[sensor]
    return np.tanh((theta - c) / w)


def prior_weights(mu=0.0, tau=1.0):
    w = np.exp(-0.5 * ((TH - mu) / tau) ** 2)
    return w / w.sum()


def _lik_grid(sensor):
    m = mean_fn(TH, sensor)
    return np.exp(-0.5 * ((Y[None, :] - m[:, None]) / S_NOISE) ** 2) / (np.sqrt(2 * np.pi) * S_NOISE)


_LIK = {}


def lik(sensor):
    if sensor not in _LIK:
        _LIK[sensor] = _lik_grid(sensor)
    return _LIK[sensor]


def d_grid(sensor, wq):
    """d(theta) = KL(p(y|theta) || q(y)) on the theta grid, with q(y) the marginal under weights wq."""
    L = lik(sensor)
    marg = wq @ L
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = L * (np.log(np.maximum(L, 1e-300)) - np.log(np.maximum(marg, 1e-300)))
    return integrand.sum(axis=1) * DY


def eig(sensor, mu=0.0, tau=1.0):
    w = prior_weights(mu, tau)
    return float(w @ d_grid(sensor, w))


def reig(sensor, eps, mu=0.0, tau=1.0):
    """REIG (Go & Isaac eq. 18) with exact d(theta), reference prior N(mu, tau^2) as
    quadrature weights (default: the standard reference prior)."""
    w = prior_weights(mu, tau)
    d = d_grid(sensor, w)
    if eps <= 0:
        return float(w @ d)
    keep = w > 1e-14
    d, lw = d[keep], np.log(w[keep])
    scale = max(np.sqrt(w[keep] @ (d - w[keep] @ d) ** 2), 1e-12)
    obj = lambda t: np.exp(t) * scale * eps + np.exp(t) * scale * logsumexp(lw - d / (np.exp(t) * scale))
    res = minimize_scalar(obj, bounds=(-15, 15), method="bounded")
    return float(min(max(-res.fun, d.min()), w[keep] @ d))


def kl_gauss(mu, tau):
    """KL( N(mu, tau^2) || N(0, 1) )."""
    return 0.5 * (tau**2 + mu**2 - 1 - 2 * np.log(tau))


def gaussian_ball(eps, n_angle=48, n_rad=6):
    """Gaussian priors N(mu, tau^2) with KL <= eps: a grid inside and on the boundary."""
    out = [(0.0, 1.0)]
    for r in np.linspace(1 / n_rad, 1, n_rad):
        for a in np.linspace(0, 2 * np.pi, n_angle, endpoint=False):
            # parametrise by (mu, log tau) direction, then scale to hit KL = r * eps
            dmu, dlt = np.cos(a), np.sin(a)
            f = lambda s: kl_gauss(s * dmu, np.exp(s * dlt)) - r * eps
            lo, hi = 0.0, 1.0
            while f(hi) < 0 and hi < 50:
                hi *= 2
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
            s = 0.5 * (lo + hi)
            out.append((s * dmu, np.exp(s * dlt)))
    return out


def gaussian_worst(sensor, eps, ball=None):
    """Worst-case EIG over Gaussian priors in the KL ball (upper bound on the true worst case)."""
    ball = ball if ball is not None else gaussian_ball(eps)
    return min(eig(sensor, mu, tau) for mu, tau in ball)


# ---------- sampling estimator (for the noisy-budget experiment) ----------

def d_samples(sensor, n1=200, n2=5, m=10, contrastive=False, rng=None):
    """Algorithm 1 step 2 for the sensor model, plain nested MC (or contrastive)."""
    rng = np.random.default_rng(rng)
    th = rng.normal(size=n1)
    mu = mean_fn(th, sensor)
    y = mu[:, None] + rng.normal(0, S_NOISE, (n1, n2))
    th_in = rng.normal(size=m)
    mu_in = mean_fn(th_in, sensor)
    ll = lambda yy, mm: -0.5 * ((yy - mm) / S_NOISE) ** 2 - np.log(np.sqrt(2 * np.pi) * S_NOISE)
    own = ll(y, mu[:, None])
    inner = ll(y[..., None], mu_in)
    if contrastive:
        inner = np.concatenate([inner, own[..., None]], axis=-1)
    log_ev = logsumexp(inner, axis=-1) - np.log(inner.shape[-1])
    return (own - log_ev).mean(axis=1)


def reig_tilt_prior(sensor, eps):
    """REIG's own minimiser: q* proportional to p * exp(-d/lam*) (from the dual).
    Returns grid weights q* and its KL to p. Evaluating the REAL EIG at q* gives
    a feasible point, hence another upper bound on the true worst case."""
    w = prior_weights()
    d = d_grid(sensor, w)
    keep = w > 1e-14
    lw = np.full_like(w, -np.inf); lw[keep] = np.log(w[keep])

    def kl_of(lam):
        lq = lw - d / lam
        lq -= logsumexp(lq[keep])
        q = np.exp(lq)
        return float(q[keep] @ (lq[keep] - lw[keep])), q

    lo, hi = -12.0, 12.0          # bisection on log lam: KL decreases as lam grows
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kl_of(np.exp(mid))[0] > eps:
            lo = mid
        else:
            hi = mid
    kl, q = kl_of(np.exp(hi))
    return q, kl


def eig_weights(sensor, wq):
    return float(wq @ d_grid(sensor, wq))


def d_exact_at(theta, sensor):
    """Exact d(theta) (grid quadrature) at arbitrary theta values, by interpolation."""
    return np.interp(theta, TH, d_grid(sensor, prior_weights()))


def d_samples_indep(sensor, n1=200, n2=1, m=20, contrastive=False, rng=None):
    """Sensor-model version with independent inner samples per theta_i (i.i.d. d_i)."""
    rng = np.random.default_rng(rng)
    th = rng.normal(size=n1)
    mu = mean_fn(th, sensor)
    y = mu[:, None] + rng.normal(0, S_NOISE, (n1, n2))
    mu_in = mean_fn(rng.normal(size=(n1, 1, m)), sensor)
    ll = lambda yy, mm: -0.5 * ((yy - mm) / S_NOISE) ** 2 - np.log(np.sqrt(2 * np.pi) * S_NOISE)
    own = ll(y, mu[:, None])
    inner = ll(y[..., None], mu_in)
    if contrastive:
        inner = np.concatenate([inner, own[..., None]], axis=-1)
    log_ev = logsumexp(inner, axis=-1) - np.log(inner.shape[-1])
    return (own - log_ev).mean(axis=1)

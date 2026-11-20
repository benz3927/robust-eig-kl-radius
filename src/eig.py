"""EIG and robust EIG (REIG) following Go & Isaac (2022), arXiv:2205.09914.

What REIG is (Sec. 5): the minimum, over priors q with KL(q || p) <= eps, of the
AFFINE (tangent) approximation I_aff(q) of EIG (eq. 11, 14). It is not the true
worst-case EIG I_true (eq. 9), but  I(p) >= I_eps >= I_true  and
|I_eps - I_true| <= eps  (eq. 15-16).

Computation (eq. 18, Algorithm 1):
    d(theta) = KL( p(y|theta) || p(y) )   (>= 0), per prior sample
    I_eps    = - inf_{lam>=0} lam*eps + lam*log E_p[exp(-d/lam)]

Baseline model (linear-Gaussian, 1-D parameter, design x):
    theta ~ N(0, s_theta^2),   y | theta, x ~ N(x * theta, s_noise^2)
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.special import logsumexp


# ---------- exact quantities for the linear-Gaussian model ----------

def eig_closed_form(x, s_theta=1.0, s_noise=1.0):
    return 0.5 * np.log1p((np.asarray(x) ** 2) * s_theta**2 / s_noise**2)


def _d_coeffs(x, s_theta=1.0, s_noise=1.0):
    """d(theta) = a + b * (theta/s_theta)^2 for the linear-Gaussian model."""
    v = s_noise**2 + x**2 * s_theta**2
    a = 0.5 * (np.log(v / s_noise**2) + s_noise**2 / v - 1.0)
    b = 0.5 * x**2 * s_theta**2 / v
    return a, b


def d_closed_form(theta, x, s_theta=1.0, s_noise=1.0):
    """KL( N(x theta, s_noise^2) || N(0, v) ), the per-theta information gain."""
    a, b = _d_coeffs(x, s_theta, s_noise)
    return a + b * (np.asarray(theta) / s_theta) ** 2


def reig_closed_form(x, eps, s_theta=1.0, s_noise=1.0):
    """Exact REIG with no Monte Carlo error at all.

    With z = theta/s_theta ~ N(0,1) and d = a + b z^2,
        E[exp(-d/lam)] = exp(-a/lam) (1 + 2b/lam)^(-1/2),
    so  I_eps = a + sup_{lam>0} [ (lam/2) log(1 + 2b/lam) - lam*eps ].
    """
    a, b = _d_coeffs(x, s_theta, s_noise)
    if eps <= 0 or b == 0:
        return a + b
    f = lambda t: -(0.5 * np.exp(t) * np.log1p(2 * b / np.exp(t)) - np.exp(t) * eps)
    res = minimize_scalar(f, bounds=(-25, 15), method="bounded")
    return a + max(-res.fun, 0.0)   # lam -> 0 limit gives min d = a


def eig_gaussian_worst_case(x, eps, s_theta=1.0, s_noise=1.0):
    """min EIG over GAUSSIAN priors N(mu, tau^2 s_theta^2) with KL to the reference <= eps.

    EIG depends only on tau, so the minimiser has mu = 0 and the smallest tau with
    0.5 (tau^2 - 1 - log tau^2) = eps. This is the worst case within the Gaussian
    family only, so it is an UPPER bound on the true worst case I_true.
    """
    if eps <= 0:
        return eig_closed_form(x, s_theta, s_noise)
    h = lambda t2: 0.5 * (t2 - 1 - np.log(t2)) - eps
    tau2 = brentq(h, 1e-300, 1.0)
    return eig_closed_form(x, s_theta * np.sqrt(tau2), s_noise)


# ---------- sampling estimators ----------

def _log_lik(y, theta, x, s_noise):
    return -0.5 * np.log(2 * np.pi * s_noise**2) - 0.5 * ((y - x * theta) / s_noise) ** 2


def d_samples_nmc(x, n1=500, n2=10, m=500, s_theta=1.0, s_noise=1.0,
                  contrastive=False, rng=None):
    """Algorithm 1, step 2: estimate d_i for each of n1 prior samples theta_i.

    For each theta_i draw n2 outcomes y_ij ~ p(y|theta_i) and average
    log p(y_ij|theta_i) - log p_hat(y_ij).

    p_hat(y) uses ONE shared set of m prior samples for all outcomes. This is
    valid but makes the errors in different d_i correlated.

    contrastive=False: plain nested MC. Biased UP for finite m (log of a mean
        underestimates log p(y)), like VNMC in the paper.
    contrastive=True:  also include theta_i itself in p_hat (as in ACE / PCE).
        Biased DOWN for finite m.
    Returns (theta, d) with d of shape (n1,).
    """
    rng = np.random.default_rng(rng)
    theta = rng.normal(0.0, s_theta, n1)
    y = x * theta[:, None] + rng.normal(0.0, s_noise, (n1, n2))
    theta_in = rng.normal(0.0, s_theta, m)
    log_lik_own = _log_lik(y, theta[:, None], x, s_noise)               # (n1, n2)
    log_lik_in = _log_lik(y[..., None], theta_in, x, s_noise)           # (n1, n2, m)
    if contrastive:
        log_lik_in = np.concatenate([log_lik_in, log_lik_own[..., None]], axis=-1)
    log_evidence = logsumexp(log_lik_in, axis=-1) - np.log(log_lik_in.shape[-1])
    return theta, (log_lik_own - log_evidence).mean(axis=1)


def log_ratio_samples_nmc(x, n=5000, m=500, s_theta=1.0, s_noise=1.0, rng=None):
    """One y per theta: per-(theta, y) terms log p(y|theta)/p_hat(y), as in eq. (4).
    Feeding THESE to reig_dual gives the joint-ambiguity criterion of eq. (20),
    not the paper's REIG. Kept for comparison only."""
    _, g = d_samples_nmc(x, n1=n, n2=1, m=m, s_theta=s_theta, s_noise=s_noise, rng=rng)
    return g


# ---------- the dual problems ----------

def _dual(d, eps, sign):
    d = np.asarray(d, dtype=float)
    n = d.size
    scale = max(np.std(d), 1e-12)

    def objective(log_lam):  # lam*eps + lam*log mean exp(sign*d/lam)
        lam = np.exp(log_lam) * scale
        return lam * eps + lam * (logsumexp(sign * d / lam) - np.log(n))

    res = minimize_scalar(objective, bounds=(-15, 15), method="bounded")
    return res.fun, float(np.exp(res.x) * scale)


def reig_dual(d, eps):
    """Algorithm 1, step 3 (eq. 21): I_eps = -inf_lam [lam*eps + lam*log mean exp(-d/lam)].
    = inf { E_q[d] : KL(q || p_hat) <= eps } over reweightings of the samples.
    The value always lies in [min d, mean d]."""
    d = np.asarray(d, dtype=float)
    if eps <= 0:
        return d.mean(), np.inf
    val, lam = _dual(d, eps, -1.0)
    return min(max(-val, d.min()), d.mean()), lam


def reig_max_dual(d, eps):
    """Risk-loving version, eq. (19): sup { E_q[d] : KL(q || p_hat) <= eps }.
    The value always lies in [mean d, max d]."""
    d = np.asarray(d, dtype=float)
    if eps <= 0:
        return d.mean(), np.inf
    val, lam = _dual(d, eps, +1.0)
    return max(min(val, d.max()), d.mean()), lam


def crossing_estimate(d_up, d_lo, hi=2.0, iters=40):
    """Crossing rule (exploratory, our own; not from the paper).

    d_up: per-theta samples from an UPPER-biased estimator (e.g. plain NMC)
    d_lo: per-theta samples from a LOWER-biased estimator (e.g. contrastive)
    As eps grows, REIG(d_up, eps) falls and REIG-max(d_lo, eps) rises.
    Pick eps* where they meet and return (eps*, common value).
    Fallbacks, shared by every experiment:
      - curves already crossed at eps = 0 (mean(d_up) <= mean(d_lo)): eps* = 0,
        estimate = average of the two means
      - no crossing by eps = hi: eps* = hi, estimate = midpoint of the two curves
    """
    d_up, d_lo = np.asarray(d_up, float), np.asarray(d_lo, float)
    f = lambda e: reig_dual(d_up, e)[0] - reig_max_dual(d_lo, e)[0]
    if f(0.0) <= 0:
        return 0.0, 0.5 * (d_up.mean() + d_lo.mean())
    if f(hi) > 0:
        return hi, 0.5 * (reig_dual(d_up, hi)[0] + reig_max_dual(d_lo, hi)[0])
    lo_e, hi_e = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo_e + hi_e)
        lo_e, hi_e = (mid, hi_e) if f(mid) > 0 else (lo_e, mid)
    e = 0.5 * (lo_e + hi_e)
    return e, 0.5 * (reig_dual(d_up, e)[0] + reig_max_dual(d_lo, e)[0])

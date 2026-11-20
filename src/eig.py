"""EIG and robust EIG (REIG) following Go & Isaac (2022), arXiv:2205.09914.

Baseline model (linear-Gaussian, 1-D parameter, design x):
    theta ~ N(0, s_theta^2),   y | theta, x ~ N(x * theta, s_noise^2)
    marginal  y | x ~ N(0, v),  v = s_noise^2 + x^2 s_theta^2

Notation from the paper:
    d(theta) = KL( p(y|theta,x) || p(y|x) )          eq. (3), always >= 0
    EIG      = E_{p(theta)}[ d(theta) ]
    REIG     = I_eps = - inf_{lam>=0} lam*eps + lam*log E_p[exp(-d/lam)]   eq. (18)
The KL ball is over the PRIOR on theta only (Sec. 2.1, 4).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


# ---------- exact quantities for the linear-Gaussian model ----------

def eig_closed_form(x, s_theta=1.0, s_noise=1.0):
    return 0.5 * np.log1p((np.asarray(x) ** 2) * s_theta**2 / s_noise**2)


def d_closed_form(theta, x, s_theta=1.0, s_noise=1.0):
    """KL( N(x theta, s_noise^2) || N(0, v) ), the per-theta information gain."""
    v = s_noise**2 + x**2 * s_theta**2
    return 0.5 * (np.log(v / s_noise**2) + (s_noise**2 + (x * theta) ** 2) / v - 1.0)


# ---------- sampling estimators ----------

def _log_lik(y, theta, x, s_noise):
    return -0.5 * np.log(2 * np.pi * s_noise**2) - 0.5 * ((y - x * theta) / s_noise) ** 2


def d_samples_nmc(x, n1=500, n2=10, m=500, s_theta=1.0, s_noise=1.0, rng=None):
    """Algorithm 1, step 2: estimate d_i for each of n1 prior samples theta_i.

    For each theta_i draw n2 outcomes y_ij ~ p(y|theta_i) and average
    log p(y_ij|theta_i) - log p_hat(y_ij), where p_hat(y) uses m fresh
    prior samples (plain nested Monte Carlo, the simplest inner estimator).
    Returns (theta, d) with d of shape (n1,).
    """
    rng = np.random.default_rng(rng)
    theta = rng.normal(0.0, s_theta, n1)
    y = x * theta[:, None] + rng.normal(0.0, s_noise, (n1, n2))
    theta_in = rng.normal(0.0, s_theta, m)
    log_lik_in = _log_lik(y[..., None], theta_in, x, s_noise)          # (n1, n2, m)
    log_evidence = logsumexp(log_lik_in, axis=-1) - np.log(m)
    log_ratio = _log_lik(y, theta[:, None], x, s_noise) - log_evidence  # (n1, n2)
    return theta, log_ratio.mean(axis=1)


def log_ratio_samples_nmc(x, n=5000, m=500, s_theta=1.0, s_noise=1.0, rng=None):
    """One y per theta: per-(theta, y) terms log p(y|theta)/p_hat(y), as in eq. (4).
    Feeding THESE to reig_dual gives the joint-ambiguity criterion of eq. (20),
    not the paper's REIG. Kept for comparison only."""
    _, g = d_samples_nmc(x, n1=n, n2=1, m=m, s_theta=s_theta, s_noise=s_noise, rng=rng)
    return g


# ---------- the dual problem ----------

def reig_dual(d, eps):
    """Algorithm 1, step 3:  I_eps = -inf_{lam>=0} [lam*eps + lam*log mean exp(-d/lam)].

    Equivalent to inf { E_q[d] : KL(q || p_hat) <= eps } for the empirical
    distribution p_hat of the samples d. Returns (I_eps, lam_star).
    lam -> infinity gives mean(d) (eps = 0); lam -> 0 gives min(d) (large eps).
    """
    d = np.asarray(d, dtype=float)
    if eps <= 0:
        return d.mean(), np.inf
    n = d.size
    scale = max(np.std(d), 1e-12)

    def objective(log_lam):  # lam*eps + lam*log mean exp(-d/lam)
        lam = np.exp(log_lam) * scale
        return lam * eps + lam * (logsumexp(-d / lam) - np.log(n))

    res = minimize_scalar(objective, bounds=(-15, 15), method="bounded")
    value = -res.fun
    value = min(max(value, d.min()), d.mean())   # dual value lies in [min d, mean d]
    return value, float(np.exp(res.x) * scale)

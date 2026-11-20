"""The two-blood-test example of Go & Isaac (2022), Sec. 3, computed exactly.

theta in {0, 1} (1 = has the condition), prior P(theta = 1) = pi.
Outcome y in {neg, pos}.
    Test A: false-negative rate 1e-14, false-positive rate 0.5
    Test B: equal false-negative and false-positive rate r, with r chosen so that
            both tests have the same EIG at pi = 0.5 (the paper says r ~ 18.4%).
"""
import numpy as np
from scipy.optimize import brentq


def lik(test, r_b):
    """Return P(y = pos | theta) for theta = 0, 1."""
    if test == "A":
        return np.array([0.5, 1.0 - 1e-14])
    return np.array([r_b, 1.0 - r_b])


def _kl_bern(a, b):
    a = np.clip(a, 1e-300, 1 - 1e-16); b = np.clip(b, 1e-300, 1 - 1e-16)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def d_values(test, pi_ref, r_b):
    """d(theta) = KL(p(y|theta) || p_ref(y)) for theta = 0, 1."""
    p_pos = lik(test, r_b)
    marg = (1 - pi_ref) * p_pos[0] + pi_ref * p_pos[1]
    return np.array([_kl_bern(p_pos[0], marg), _kl_bern(p_pos[1], marg)])


def eig(test, pi, r_b):
    return (1 - pi) * d_values(test, pi, r_b)[0] + pi * d_values(test, pi, r_b)[1]


def calibrate_rb(pi=0.5):
    target = eig("A", pi, 0.0)
    return brentq(lambda r: eig("B", pi, r) - target, 1e-6, 0.499)


def kl_bern_prior(q, p):
    return _kl_bern(q, p)


def ball_interval(pi_ref, eps):
    """All q with KL(Bern(q) || Bern(pi_ref)) <= eps form an interval [lo, hi]."""
    if eps <= 0:
        return pi_ref, pi_ref
    f = lambda q: kl_bern_prior(q, pi_ref) - eps
    lo = brentq(f, 1e-15, pi_ref) if f(1e-15) > 0 else 0.0
    hi = brentq(f, pi_ref, 1 - 1e-15) if f(1 - 1e-15) > 0 else 1.0
    return lo, hi


def reig(test, pi_ref, eps, r_b):
    """REIG: min over the ball of E_q[d_ref]. Linear in q, so an endpoint wins."""
    d = d_values(test, pi_ref, r_b)
    lo, hi = ball_interval(pi_ref, eps)
    return min((1 - q) * d[0] + q * d[1] for q in (lo, hi))


def true_worst(test, pi_ref, eps, r_b, grid=4001):
    """I_true: min over the ball of the real EIG(q) (grid search on the interval)."""
    lo, hi = ball_interval(pi_ref, eps)
    qs = np.linspace(lo, hi, grid)
    return min(eig(test, q, r_b) for q in qs)

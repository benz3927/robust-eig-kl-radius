"""Exp 10 - Symbolic (exact) check of the algebra behind the expansion in Theorem 1,
plus a second-order correction, checked against the linear-Gaussian closed form.

The minimiser of E_q[d] over the KL ball is the exponential tilt q_t ∝ p e^{-t d}
(this is where the dual's lambda comes from, t = 1/lambda). With the cumulant
generating function psi(t) = log E_p exp(-t (d - mu)) = s^2 t^2/2 - k3 t^3/6 + k4 t^4/24 - ...:
    E_{q_t}[d] = mu - psi'(t),     KL(q_t || p) = t psi'(t) - psi(t).
Solve KL = eps for t as a series in sqrt(eps) and substitute.
"""
import sympy as sp
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eig import reig_closed_form, eig_closed_form, _d_coeffs

t, e, s, k3, k4, mu = sp.symbols("t epsilon sigma kappa3 kappa4 mu", positive=True)
r = sp.symbols("r", positive=True)           # r = sqrt(eps)
psi = s**2 * t**2 / 2 - k3 * t**3 / 6 + k4 * t**4 / 24
dpsi = sp.diff(psi, t)
KL = sp.expand(t * dpsi - psi)
mean_q = mu - dpsi
print("KL(q_t||p)   =", sp.simplify(KL))
print("E_{q_t}[d]   =", sp.expand(mean_q))

# ansatz t = c1 r + c2 r^2 + c3 r^3, solve KL = r^2 order by order
c1, c2, c3 = sp.symbols("c1 c2 c3")
tt = c1 * r + c2 * r**2 + c3 * r**3
eq = sp.expand(sp.series(KL.subs(t, tt) - r**2, r, 0, 5).removeO())
sol1 = sp.solve(sp.Eq(eq.coeff(r, 2), 0), c1)
sol1 = [x for x in sol1 if x.is_positive is not False][0]
sol2 = sp.solve(sp.Eq(eq.coeff(r, 3).subs(c1, sol1), 0), c2)[0]
sol3 = sp.solve(sp.Eq(eq.coeff(r, 4).subs({c1: sol1, c2: sol2}), 0), c3)[0]
t_series = (c1 * r + c2 * r**2 + c3 * r**3).subs({c1: sol1, c2: sol2, c3: sol3})
val = sp.series(mean_q.subs(t, t_series), r, 0, 4).removeO()
val = sp.simplify(sp.expand(val))
print("t*(eps)      =", sp.simplify(t_series), "   (r = sqrt(eps))")
print("REIG(eps)    =", sp.collect(sp.expand(val), r), "+ O(eps^2)")

first = mu - s * sp.sqrt(2) * r
assert sp.simplify(sp.expand(val).coeff(r, 1) - (-sp.sqrt(2) * s)) == 0, "first-order term wrong"
print("CHECK: first-order term equals -sqrt(2 eps) sigma  ->  OK")
z, N = sp.symbols("z N", positive=True)
assert sp.simplify(sp.sqrt(2) * s * sp.sqrt(z**2 / (2 * N)) - z * s / sp.sqrt(N)) == 0
print("CHECK: eps = z^2/(2N) gives penalty z sigma / sqrt(N)  ->  OK")

# numerical check of the second-order term on the linear-Gaussian model, d = a + b Z^2
coef2 = sp.expand(val).coeff(r, 2)
print("second-order coefficient (of eps):", sp.simplify(coef2))
f2 = sp.lambdify((s, k3), coef2)
print("\nlinear-Gaussian check (exact REIG vs 1st and 2nd order):")
for x in [1.0, 2.0]:
    a, b = _d_coeffs(x)
    m, sig, kap3 = a + b, np.sqrt(2) * b, 8 * b**3        # cumulants of a + b chi^2_1
    for eps in [1e-3, 1e-2, 5e-2]:
        ex = reig_closed_form(x, eps)
        o1 = m - sig * np.sqrt(2 * eps)
        o2 = o1 + f2(sig, kap3) * eps
        print(f"  x={x} eps={eps:<6} exact={ex:.6f}  1st-order err={ex-o1:+.2e}  2nd-order err={ex-o2:+.2e}")

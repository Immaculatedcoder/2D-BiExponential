import numpy as np

from config import (
    TI,
    TE,
    TRUE_PARAMS,
    THETA_LOWER,
    THETA_UPPER,
    THETA0,
    SIGMA,
    RANDOM_SEED
)

from model import biexponential_2d

from data import add_gaussian_noise

from saddle2 import (
    build_theta_preconditioner,
    solve_saddle_preconditioned,
    saddle_diagnostics
)


# ============================================================
# 1. Generate clean data
# ============================================================

D_clean = biexponential_2d(
    TI,
    TE,
    TRUE_PARAMS
)


# ============================================================
# 2. Add Gaussian noise
#
# D_noisy = D_clean + epsilon
#
# epsilon ~ N(0, sigma^2)
# ============================================================

D_noisy = add_gaussian_noise(
    D_clean,
    sigma=SIGMA,
    seed=RANDOM_SEED
)


# Vectorize using the SAME convention used everywhere else
d_noisy = D_noisy.ravel(
    order="F"
)


# ============================================================
# 3. True nonlinear parameters
# ============================================================

theta_true = np.array([
    TRUE_PARAMS["T11"],
    TRUE_PARAMS["T12"],
    TRUE_PARAMS["T21"],
    TRUE_PARAMS["T22"]
], dtype=float)


# ============================================================
# 4. Build theta preconditioner
# ============================================================

P_theta = build_theta_preconditioner(
    THETA_LOWER,
    THETA_UPPER
)

P_mu = np.eye(6)

p_lam = 1.0


print(
    "\n================================"
)

print(
    "NOISY-DATA EXPERIMENT"
)

print(
    "================================"
)

print("\nSigma:")
print(SIGMA)

print("\nRandom seed:")
print(RANDOM_SEED)

print("\nInitial theta:")
print(THETA0)

print("\nTrue theta:")
print(theta_true)

print("\nTheta preconditioner:")
print(P_theta)


# ============================================================
# 5. Solve noisy inverse problem
# ============================================================

solution = solve_saddle_preconditioned(

    THETA0,

    np.zeros(6),

    0.0,

    TI,
    TE,

    d_noisy,

    THETA_LOWER,
    THETA_UPPER,

    t_span=(
        0.0,
        10000.0
    ),

    P_theta=P_theta,

    P_mu=P_mu,

    p_lam=p_lam,

    gamma_theta=1.0,

    gamma_mu=1.0,

    gamma_lam=1.0,

    method="BDF",

    rtol=1e-6,

    atol=1e-8,

    max_step=20.0,

    steady_tol=1e-5
)


# ============================================================
# 6. Diagnostics
# ============================================================

diagnostics = saddle_diagnostics(

    solution,

    TI,
    TE,

    d_noisy,

    THETA_LOWER,
    THETA_UPPER,

    preconditioned=True,

    P_theta=P_theta,

    P_mu=P_mu,

    p_lam=p_lam,

    gamma_theta=1.0,

    gamma_mu=1.0,

    gamma_lam=1.0
)


# ============================================================
# 7. Estimated parameters
# ============================================================

theta_est = diagnostics["theta"]

c_est = diagnostics["c"]

beta1_est = diagnostics["beta1"]
beta2_est = diagnostics["beta2"]


# ============================================================
# 8. Errors in theta
# ============================================================

theta_error = (
    theta_est
    - theta_true
)

theta_relative_error = (
    theta_error
    / theta_true
)

theta_percent_error = (
    100.0
    * theta_relative_error
)


# ============================================================
# 9. Residual objective
#
# F = 0.5 ||r||^2
# ============================================================

residual_norm = diagnostics[
    "residual_norm"
]

objective = (
    0.5
    * residual_norm**2
)


# ============================================================
# 10. Expected noise level
#
# For N independent Gaussian observations,
#
# E[ ||epsilon||^2 ] = N sigma^2
#
# so approximately
#
# E[ ||epsilon|| ] ~ sigma sqrt(N)
#
# and
#
# E[ 0.5 ||epsilon||^2 ]
#     = 0.5 N sigma^2
# ============================================================

N = d_noisy.size

expected_noise_norm = (
    SIGMA
    * np.sqrt(N)
)

expected_objective = (
    0.5
    * N
    * SIGMA**2
)


# ============================================================
# 11. Print solver status
# ============================================================

print(
    "\n================================"
)

print(
    "SOLVER STATUS"
)

print(
    "================================"
)

print(
    "Success:",
    solution.success
)

print(
    "Message:",
    solution.message
)

print(
    "Final time:",
    solution.t[-1]
)

print(
    "Stored time points:",
    len(solution.t)
)

print(
    "Function evaluations:",
    solution.nfev
)

print(
    "Jacobian evaluations:",
    solution.njev
)

print(
    "LU decompositions:",
    solution.nlu
)

print(
    "\nB-event:"
)

print(
    solution.t_events[0]
)

print(
    "\nSteady-state event:"
)

print(
    solution.t_events[1]
)


# ============================================================
# 12. Print estimates
# ============================================================

print(
    "\n================================"
)

print(
    "PARAMETER ESTIMATES"
)

print(
    "================================"
)

print(
    "\nTrue theta:"
)

print(
    theta_true
)

print(
    "\nEstimated theta:"
)

print(
    theta_est
)

print(
    "\nTheta error:"
)

print(
    theta_error
)

print(
    "\nTheta percent error:"
)

print(
    theta_percent_error
)


print(
    "\nEstimated c:"
)

print(
    c_est
)


print(
    "\nEstimated beta:"
)

print(
    "beta1 =",
    beta1_est
)

print(
    "beta2 =",
    beta2_est
)


# ============================================================
# 13. Constraint diagnostics
# ============================================================

print(
    "\n================================"
)

print(
    "CONSTRAINT DIAGNOSTICS"
)

print(
    "================================"
)

print(
    "\nFinal mu:"
)

print(
    diagnostics["mu"]
)

print(
    "\nFinal lambda:"
)

print(
    diagnostics["lambda"]
)

print(
    "\nInequality residuals g:"
)

print(
    diagnostics["g"]
)

print(
    "\nEquality residual h:"
)

print(
    diagnostics["h"]
)


# ============================================================
# 14. Residual diagnostics
# ============================================================

print(
    "\n================================"
)

print(
    "RESIDUAL DIAGNOSTICS"
)

print(
    "================================"
)

print(
    "\nNumber of observations:"
)

print(
    N
)

print(
    "\nNoise sigma:"
)

print(
    SIGMA
)

print(
    "\nFinal residual norm:"
)

print(
    residual_norm
)

print(
    "\nExpected noise norm "
    "(approx sigma*sqrt(N)):"
)

print(
    expected_noise_norm
)

print(
    "\nFinal objective "
    "0.5*||r||^2:"
)

print(
    objective
)

print(
    "\nExpected noise objective "
    "0.5*N*sigma^2:"
)

print(
    expected_objective
)


# ============================================================
# 15. Stationarity diagnostics
# ============================================================

print(
    "\n================================"
)

print(
    "STATIONARITY"
)

print(
    "================================"
)

print(
    "\nFinal grad_theta:"
)

print(
    diagnostics["grad_theta"]
)

print(
    "\nActual preconditioned "
    "flow RHS norm:"
)

print(
    diagnostics[
        "flow_rhs_norm"
    ]
)

print(
    "\nPreconditioner-independent "
    "stationarity norm:"
)

print(
    diagnostics[
        "stationarity_norm"
    ]
)


# ============================================================
# 16. B diagnostics
# ============================================================

print(
    "\n================================"
)

print(
    "REDUCED MATRIX B"
)

print(
    "================================"
)

print(
    "\nMinimum eigenvalue of B:"
)

print(
    diagnostics[
        "B_min_eigenvalue"
    ]
)

print(
    "\nCondition number of B:"
)

print(
    diagnostics[
        "B_condition_number"
    ]
)
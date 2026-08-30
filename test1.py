import numpy as np

from config import TI, TE, TRUE_PARAMS, THETA0, SIGMA, RANDOM_SEED, THETA_LOWER, THETA_UPPER
from model import biexponential_2d
from data import add_gaussian_noise
from varpro import build_design_matrix, build_design_matrix_derivatives, solve_linear_coefficients
from optimizer import solve_theta

# -----------
# 1. Generate clean data
# -----------

D_clean = biexponential_2d(TI, TE, TRUE_PARAMS)

# -----------
# 2. Add Gaussian noise
# -----------

D_noisy = add_gaussian_noise(D_clean, SIGMA, RANDOM_SEED)
d_noisy = D_noisy.ravel(order="F")

# -----------
# 3. recover the Nonlinea  
# -----------
result = solve_theta(
    THETA0, TI, TE, d_noisy,THETA_LOWER, THETA_UPPER
)

theta_est = result.x
# -----------
# 4. recover the linear
# -----------

Phi_est = build_design_matrix(TI, TE, theta_est)
c_est = solve_linear_coefficients(Phi_est, d_noisy)

k1_est = c_est[0]
beta1_est = c_est[1] / c_est[0]

k2_est = c_est[2]
beta2_est = c_est[3] / c_est[2]

# -----
# 5. Print results
# -------
print(theta_est)
print("k1    =", k1_est)
print("beta1 =", beta1_est)
print("k2    =", k2_est)
print("beta2 =", beta2_est)
# -----------------------

theta_true = np.array([
    TRUE_PARAMS["T11"],
    TRUE_PARAMS["T12"],
    TRUE_PARAMS["T21"],
    TRUE_PARAMS["T22"]
])

theta_abs_error = np.abs(theta_est - theta_true)

theta_rel_error = (
    theta_abs_error / np.abs(theta_true)
) * 100.0

physical_true = np.array([
    TRUE_PARAMS["k1"],
    TRUE_PARAMS["beta1"],
    TRUE_PARAMS["k2"],
    TRUE_PARAMS["beta2"]
])

physical_est = np.array([
    k1_est,
    beta1_est,
    k2_est,
    beta2_est
])

physical_abs_error = np.abs(
    physical_est - physical_true
)

physical_rel_error = (
    physical_abs_error / np.abs(physical_true)
) * 100.0

print("\nTheta absolute error:")
print(theta_abs_error)

print("\nTheta relative error (%):")
print(theta_rel_error)

print("\nPhysical parameter absolute error:")
print(physical_abs_error)

print("\nPhysical parameter relative error (%):")
print(physical_rel_error)

def check_design_matrix_derivatives(TI, TE, theta):
    """
    Compare analytic derivatives of Phi with
    centered finite-difference approximations.
    """

    analytic_derivatives = build_design_matrix_derivatives(
        TI,
        TE,
        theta
    )

    names = ["T11", "T12", "T21", "T22"]

    for j, name in enumerate(names):

        # Scale finite-difference step to parameter size
        h = 1e-6 * max(1.0, abs(theta[j]))

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[j] += h
        theta_minus[j] -= h

        Phi_plus = build_design_matrix(
            TI,
            TE,
            theta_plus
        )

        Phi_minus = build_design_matrix(
            TI,
            TE,
            theta_minus
        )

        finite_difference = (
            Phi_plus - Phi_minus
        ) / (2.0 * h)

        analytic = analytic_derivatives[j]

        absolute_error = np.linalg.norm(
            analytic - finite_difference
        )

        relative_error = (
            absolute_error
            / np.linalg.norm(finite_difference)
        )

        print(f"{name}:")
        print(f"  absolute error = {absolute_error:.6e}")
        print(f"  relative error = {relative_error:.6e}")
check_design_matrix_derivatives(
        TI,
        TE,
        THETA0
    )
if __name__ == "__main__":

    derivatives = build_design_matrix_derivatives(
        TI,
        TE,
        THETA0
    )

    for dPhi in derivatives:
        print(dPhi.shape)

    
import numpy as np

from scipy.optimize import (minimize, lsq_linear, LinearConstraint, NonlinearConstraint, Bounds)

from varpro import (build_design_matrix, build_design_matrix_derivatives)


def unpack_x(x):
    """
    Split the full SLSQP vector
        x = [c, theta]

        into
            c = [k1, k1_hat, k2, k2_hat]
            theta = [T11, T12, T21, T22]
    """

    x = np.asarray(x, dtype=float)

    c = x[:4]
    theta = x[4:]

    return c, theta

def pack_x(c, theta):
    """
        combine c and theta into the full SLSQP vector

        x = [c, theta]
    """

    c = np.asarray(c, dtype=float)
    theta = np.asarray(theta, dtype=float)

    return np.concatenate([
        c,
        theta
    ])

def build_initial_c(TI, TE, d, theta0, beta0=1.0):
    """
    Construct c0 such that
        k1_hat = beta0 * k1
        k2_hat = beta0 * k2
    """

    beta0 = float(beta0)

    if beta0 < 0.0 or beta0 > 2.0:
        raise ValueError(
            "beta0 must line in [0,2]"
        )

    Phi0 = build_design_matrix(TI, TE, theta0)
    # Phi(theta0)c0 = k1*(phi1 + beta0*phi2) + k2*(phi3 + beta0*phi4)

    M = np.column_stack([
        Phi0[:, 0]+ beta0 * Phi0[:, 1],
        Phi0[:, 2]+ beta0 * Phi0[:, 3]
    ])

    # solve for postive amplitudes
    fit = lsq_linear(
        M,
        d,
        bounds=(0.0, np.inf),
        method='trf',
        tol=1e-12
    )

    if not fit.success:
        raise RuntimeError(
            f"Initial amplitude solve failed: {fit.message}"
        )

    k1, k2 = fit.x
    eps_k = 1e-8
    k1 = max(k1, eps_k)
    k2 = max(k2, eps_k)

    c0 = np.array([
        k1,
        beta0*k1,
        k2,
        beta0*k2
    ])

    return c0

def objective(x, TI, TE, d):
    """
    The full objective
        F(x) = F(c, theta) = 0.5 * ||Phi(theta)c - d||_2^2
        x = [c, theta]
    """

    # Get c, theta from x
    c, theta = unpack_x(x)

    # Build Phi(theta)
    Phi = build_design_matrix(TI, TE, theta)

    # Residual
    residual = Phi @ c - d
    value = 0.5 * np.dot(residual, residual)

    return value

def objective_gradient(x, TI, TE, d):
    """
    ∇_x F  = [∇_c F
              ∇_θ F]
    x = [c, θ]
    """
    # ∇_c F
    c, theta = unpack_x(x)
    Phi = build_design_matrix(TI, TE, theta)
    residual = Phi @ c - d

    grad_c = Phi.T @ residual

    # ∇_θ F
    (dPhi_dT11, dPhi_dT12, dPhi_dT21, dPhi_dT22) = build_design_matrix_derivatives(TI, TE, theta)

    dF_dT11 = residual @ (dPhi_dT11 @ c)
    dF_dT12 = residual @ (dPhi_dT12 @ c)
    dF_dT21 = residual @ (dPhi_dT21 @ c)
    dF_dT22 = residual @ (dPhi_dT22 @ c)

    grad_theta = np.array([
        dF_dT11,
        dF_dT12,
        dF_dT21,
        dF_dT22
    ])

    gradient = np.concatenate([
        grad_c,
        grad_theta
    ])

    return gradient

def build_linear_constraints():
    """
    k1_hat - 2k1 <= 0
    k2_hat - 2k2 <= 0
    """

    C = np.zeros((2,8), dtype=float)
    # k1_hat - 2k1 <= 0
    C[0, 0] = -2.0
    C[0, 1] = 1.0

    # k2_hat - 2k2 <= 0
    C[1, 2] = -2.0
    C[1, 3] = 1.0

    constraint = LinearConstraint(
        C,
        lb=[-np.inf, -np.inf],
        ub=[0.0, 0.0]
    )

    return constraint

def equality_constraint(x):
    """
        Equality constraint
        h(c) = k1_hat * k2 - k2_hat * k1 = 0
    """
    c, theta = unpack_x(x)

    k1 = c[0]
    k1_hat = c[1]
    k2 = c[2]
    k2_hat = c[3]

    h = (k1_hat * k2 - k1 * k2_hat)

    return h

def equality_jacobian(x):
    """
    Jacobian of

        h(c) = k1_hat*k2 - k1*k2_hat.
    """

    c, theta = unpack_x(x)

    k1 = c[0]
    k1_hat = c[1]
    k2 = c[2]
    k2_hat = c[3]

    jac = np.array([
        -k2_hat,
        k2,
        k1_hat,
        -k1,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    return jac



def build_bounds(theta_lower, theta_upper, eps_k = 1e-8):
    """
        x_lb <= x <= x_ub
    """

    theta_lower=np.asarray(theta_lower, dtype=float)
    theta_upper=np.asarray(theta_upper, dtype=float)

    lower = np.concatenate([
        np.array([
            eps_k,
            0.0,
            eps_k,
            0.0
        ]),
        theta_lower
    ])

    upper = np.concatenate([
            np.array([
                np.inf,
                np.inf,
                np.inf,
                np.inf
            ]),
            theta_upper
        ])

    bounds = Bounds(lower, upper)

    return bounds

# -------------------------------------------------------------------
# --------------SLSQP------------------------------------------------
# -------------------------------------------------------------------

def solve_scipy(
        TI,
        TE,
        d,
        theta0,
        theta_lower,
        theta_upper,
        beta0=1.0
):
    # Starting point
    c0 = build_initial_c(TI, TE, d, theta0, beta0)
    x0 = pack_x(c0, theta0)

    # Constraints
    linear_constraint = (build_linear_constraints())

    equality = NonlinearConstraint(
        equality_constraint,
        lb=0.0,
        ub=0.0,
        jac=equality_jacobian
    )

    bounds = build_bounds(theta_lower, theta_upper)
    result = minimize(
        objective,
        x0,
        args = (TI, TE, d),
        jac = objective_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints = [
            linear_constraint,
            equality
        ],
        options = {
            "maxiter": 3000,
            "ftol": 1e-12,
            "disp": True
        }
    )

    # Extract Solution from results
    c_star, theta_star = unpack_x(result.x)

    k1 = c_star[0]
    k1_hat = c_star[1]

    k2 = c_star[2]
    k2_hat = c_star[3]

    beta1 = k1_hat / k1
    beta2 = k2_hat / k2

    Phi_star = build_design_matrix(
        TI,
        TE,
        theta_star
    )

    residual = (
        Phi_star @ c_star
        - d
    )

    objective_star = (
        0.5
        * np.dot(
            residual,
            residual
        )
    )

    h_star = (
        k1_hat * k2
        - k1 * k2_hat
    )

    g1 = (
        k1_hat
        - 2.0 * k1
    )

    g2 = (
        k2_hat
        - 2.0 * k2
    )

    diagnostics = {
        "success": result.success,
        "message": result.message,
        "nit": result.nit,
        "objective": objective_star,
        "h": h_star,
        "g1": g1,
        "g2": g2
    }

    return (
        c_star,
        theta_star,
        beta1,
        beta2,
        diagnostics,
        result
    )
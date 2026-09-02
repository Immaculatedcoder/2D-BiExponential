import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import lsq_linear, minimize_scalar

def build_design_matrix(TI, TE, theta):

    """
    Build the VarPro design matrix Phi(theta)

    theta = [T11, T12, T21, T22]
    """

    T11, T12, T21, T22 = theta

    TI_grid, TE_grid = np.meshgrid(
        TI,
        TE,
        indexing="ij"
    )

    phi1 = np.exp(-TE_grid / T21)

    phi2 = (
        -np.exp(-TI_grid / T11) * np.exp(-TE_grid / T21)
    )

    phi3 = np.exp(-TE_grid / T22)

    phi4 = (
        -np.exp(-TI_grid / T12) * np.exp(-TE_grid / T22)
    )

    Phi = np.column_stack([
        phi1.ravel(order="F"),
        phi2.ravel(order="F"),
        phi3.ravel(order="F"),
        phi4.ravel(order="F")
    ])

    return Phi

def build_design_matrix_derivatives(TI, TE, theta):
    """
        d Phi/d theta
    """
    T11, T12, T21, T22 = theta

    TI_grid, TE_grid = np.meshgrid(
        TI,
        TE,
        indexing="ij"
    )

    E11 = np.exp(-TI_grid / T11)
    E12 = np.exp(-TI_grid / T12)

    E21 = np.exp(-TE_grid / T21)
    E22 = np.exp(-TE_grid / T22)

    zero = np.zeros_like(TI_grid)

    # --------------------------------------------------
    # dPhi / dT11
    # --------------------------------------------------
    dphi2_dT11 = (
        -(TI_grid / T11**2)
        * E11
        * E21
    )

    dPhi_dT11 = np.column_stack([
        zero.ravel(order="F"),
        dphi2_dT11.ravel(order="F"),
        zero.ravel(order="F"),
        zero.ravel(order="F")
    ])

    # --------------------------------------------------
    # dPhi / dT12
    # --------------------------------------------------

    dphi4_dT12 = (
        -(TI_grid / T12**2)
        * E12
        * E22
    )

    dPhi_dT12 = np.column_stack([
        zero.ravel(order="F"),
        zero.ravel(order="F"),
        zero.ravel(order="F"),
        dphi4_dT12.ravel(order="F")
    ])

    # --------------------------------------------------
    # dPhi / dT21
    # --------------------------------------------------

    dphi1_dT21 = (
        (TE_grid / T21**2)
        * E21
    )

    dphi2_dT21 = (
        -(TE_grid / T21**2)
        * E11
        * E21
    )

    dPhi_dT21 = np.column_stack([
        dphi1_dT21.ravel(order="F"),
        dphi2_dT21.ravel(order="F"),
        zero.ravel(order="F"),
        zero.ravel(order="F")
    ])

    # --------------------------------------------------
    # dPhi / dT22
    # --------------------------------------------------

    dphi3_dT22 = (
        (TE_grid / T22**2)
        * E22
    )

    dphi4_dT22 = (
        -(TE_grid / T22**2)
        * E12
        * E22
    )

    dPhi_dT22 = np.column_stack([
        zero.ravel(order="F"),
        zero.ravel(order="F"),
        dphi3_dT22.ravel(order="F"),
        dphi4_dT22.ravel(order="F")
    ])

    return (
        dPhi_dT11,
        dPhi_dT12,
        dPhi_dT21,
        dPhi_dT22
    )


# ----- Test -------
# from config import *
# c_true = np.array([
#     TRUE_PARAMS["k1"],
#     TRUE_PARAMS["k1"]*TRUE_PARAMS["beta1"],
#     TRUE_PARAMS["k2"],
#     TRUE_PARAMS["k2"]*TRUE_PARAMS["beta2"]
# ])

# theta_true = np.array([
#     TRUE_PARAMS["T11"],
#     TRUE_PARAMS["T12"],
#     TRUE_PARAMS["T21"],
#     TRUE_PARAMS["T22"]
# ])

# from data import *
# D_clean = generate_clean_data(TI, TE, TRUE_PARAMS)
# d_clean = D_clean.ravel(order="F")

# Phi = build_design_matrix(TI, TE, theta_true)

# d_varpro = Phi @ c_true

# error = np.linalg.norm(d_clean - d_varpro)
# print(error)

# ---------------

# def solve_linear_coefficients(Phi, d):
    """
    Solve the constrained linear VarPro 
        min_c ||Phi c - d||_2^2

        subject to the conditions
        c0 >=0, c1 >= 0, c2 >=0, c3>= 0

        c1 <= 2*c0
        c3 <= 2*c2

        where 
            c = [k1, k1_hat, k2, k2_hat]
        
        and 
            k1_hat = k1 * beta1
            k2_hat = k2 * beta2

    Parameters
    ------------
    Phi : ndarray, shape (m,4)
    d: ndarray, shape (m,)

    Returns
    --------
    c: ndarray, shape (4,)
    """

    # For fixed theta
    # Objective function 
    def objective(c):
        residual = Phi @ c - d
        return 0.5 * np.dot(residual, residual)

    # The gradient with respect to c
    def gradient(c):
        residual = Phi @ c - d
        return Phi.T @ residual

    # Initial guess

    c0, *_ = np.linalg.lstsq(
        Phi,
        d,
        rcond=None
    )

    # Ensure that starting point is in feasible set
    c0 = np.maximum(c0, 0.0)
    c0[1] = min(c0[1], 2.0*c0[0])
    c0[3] = min(c0[3], 2.0*c0[2])

    # -------------------
    # Bounds: c_i >= 0
    # -------------------

    eps_k = 1e-8
    bounds = Bounds(
        lb=[eps_k,0.0,eps_k,0.0],
        ub=[np.inf, np.inf, np.inf, np.inf]
    )

    # --------------------
    # beta constants
    # 
    #  c1 <= 2*c0
    #  c3 <= 2*c2
    # Since SLSQP uses constraints >= 0
    # so we have
    # 2*c0 - c1 >= 0
    # 2*c2 - c3 >= 0

    constraints = [
        {
            "type": "ineq",
            "fun" : lambda c : 2.0 * c[0] - c[1],
            "jac" : lambda c : np.array([2.0, -1.0, 0.0, 0.0])
        },
        {
            "type": "ineq",
            "fun" : lambda c: 2.0 * c[2] - c[3],
            "jac" : lambda c : np.array([0.0, 0.0, 2.0, -1.0])
        }
    ]

    # --------------
    # Solve constrained problem
    # -----------------

    result = minimize(
        objective,
        c0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f"Linear SLSQP solve failed : {result.message}"
        )

    
    return result.x

def solve_linear_coefficients(
    Phi,
    d,
    beta_lower=0.0,
    beta_upper=2.0,
    eps_k=1e-8
):
    """
    Solve the equality-constrained inner VarPro problem

        min_c  0.5 * ||Phi c - d||_2^2

    subject to

        k1 > 0,
        k2 > 0,

        0 <= k1_hat <= 2*k1,
        0 <= k2_hat <= 2*k2,

        k1_hat*k2 - k1*k2_hat = 0.

    The equality constraint implies

        beta1 = beta2 = beta,

    where

        k1_hat = k1 * beta,
        k2_hat = k2 * beta.

    Therefore parameterize

        c = [
            k1,
            k1*beta,
            k2,
            k2*beta
        ].

    For fixed beta, the model is linear in k1 and k2:

        Phi c
        =
        k1 * (Phi[:,0] + beta*Phi[:,1])
        +
        k2 * (Phi[:,2] + beta*Phi[:,3]).

    Hence the inner constrained problem reduces to

        min_{beta in [0,2]}
            min_{k1,k2 >= eps_k}
                0.5 * ||M(beta) k - d||_2^2.

    Parameters
    ----------
    Phi : ndarray, shape (m,4)
        Standard VarPro design matrix.

    d : ndarray, shape (m,)
        Vectorized data.

    beta_lower : float
        Lower bound on the common beta.

    beta_upper : float
        Upper bound on the common beta.

    eps_k : float
        Small positive lower bound for k1 and k2.

    Returns
    -------
    c : ndarray, shape (4,)
        Optimal coefficient vector

            [k1, k1_hat, k2, k2_hat]

        satisfying beta1 = beta2 by construction.
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    Phi = np.asarray(
        Phi,
        dtype=float
    )

    d = np.asarray(
        d,
        dtype=float
    ).reshape(-1)

    if Phi.ndim != 2 or Phi.shape[1] != 4:
        raise ValueError(
            "Phi must have shape (m,4)."
        )

    if Phi.shape[0] != d.size:
        raise ValueError(
            "Phi and d have incompatible dimensions."
        )

    if not np.all(np.isfinite(Phi)):
        raise ValueError(
            "Phi must contain only finite values."
        )

    if not np.all(np.isfinite(d)):
        raise ValueError(
            "d must contain only finite values."
        )

    if eps_k <= 0.0:
        raise ValueError(
            "eps_k must be positive."
        )

    beta_lower = float(beta_lower)
    beta_upper = float(beta_upper)

    if beta_lower < 0.0:
        raise ValueError(
            "beta_lower must be nonnegative."
        )

    if beta_upper <= beta_lower:
        raise ValueError(
            "beta_upper must be larger than beta_lower."
        )

    # --------------------------------------------------
    # 2. Solve k1,k2 for a fixed common beta
    # --------------------------------------------------

    def solve_amplitudes(beta):
        """
        For fixed beta solve

            min_{k1,k2 >= eps_k}
                0.5 ||M(beta) k - d||^2
        """

        beta = float(beta)

        # --------------------------------------------------
        # This is the same structural idea used in
        # solver_scipy.py::_initial_c.
        #
        # Phi columns are
        #
        #   phi1
        #   phi2 = -E11*E21
        #   phi3
        #   phi4 = -E12*E22
        #
        # Therefore
        #
        #   phi1 + beta*phi2
        #
        # = (1 - beta*E11)*E21
        #
        # and similarly for compartment 2.
        # --------------------------------------------------

        amplitude_matrix = np.column_stack([
            Phi[:, 0]
            + beta * Phi[:, 1],

            Phi[:, 2]
            + beta * Phi[:, 3]
        ])

        amplitude_fit = lsq_linear(
            amplitude_matrix,
            d,
            bounds=(
                np.array(
                    [eps_k, eps_k],
                    dtype=float
                ),
                np.array(
                    [np.inf, np.inf],
                    dtype=float
                )
            ),
            method="trf",
            tol=1e-12,
            lsmr_tol="auto",
            max_iter=500
        )

        if not amplitude_fit.success:
            raise RuntimeError(
                "Amplitude least-squares solve failed: "
                f"{amplitude_fit.message}"
            )

        k = np.asarray(
            amplitude_fit.x,
            dtype=float
        )

        residual = (
            amplitude_matrix @ k
            - d
        )

        objective = (
            0.5
            * float(
                residual @ residual
            )
        )

        return objective, k

    # --------------------------------------------------
    # 3. Reduced one-dimensional objective in beta
    # --------------------------------------------------

    def beta_objective(beta):

        objective, _ = (
            solve_amplitudes(beta)
        )

        return objective

    # --------------------------------------------------
    # 4. Optimize common beta
    # --------------------------------------------------

    beta_result = minimize_scalar(
        beta_objective,
        bounds=(
            beta_lower,
            beta_upper
        ),
        method="bounded",
        options={
            "xatol": 1e-12,
            "maxiter": 500
        }
    )

    if not beta_result.success:
        raise RuntimeError(
            "Common-beta optimization failed: "
            f"{beta_result.message}"
        )

    # --------------------------------------------------
    # 5. Compare interior solution with beta endpoints
    #
    # minimize_scalar("bounded") searches the interior.
    # Explicitly checking both endpoints makes the
    # bounded problem complete.
    # --------------------------------------------------

    candidates = [
        (
            float(beta_result.fun),
            float(beta_result.x)
        )
    ]

    objective_lower, _ = (
        solve_amplitudes(
            beta_lower
        )
    )

    candidates.append(
        (
            objective_lower,
            beta_lower
        )
    )

    objective_upper, _ = (
        solve_amplitudes(
            beta_upper
        )
    )

    candidates.append(
        (
            objective_upper,
            beta_upper
        )
    )

    _, beta_star = min(
        candidates,
        key=lambda item: item[0]
    )

    # --------------------------------------------------
    # 6. Recover optimal k1,k2
    # --------------------------------------------------

    _, k_star = solve_amplitudes(
        beta_star
    )

    k1 = float(
        k_star[0]
    )

    k2 = float(
        k_star[1]
    )

    # --------------------------------------------------
    # 7. Recover c
    #
    # c = [k1, k1_hat, k2, k2_hat]
    #
    # with
    #
    # k1_hat = beta*k1
    # k2_hat = beta*k2
    # --------------------------------------------------

    c = np.array([
        k1,
        beta_star * k1,
        k2,
        beta_star * k2
    ])

    # --------------------------------------------------
    # 8. Numerical sanity checks
    # --------------------------------------------------

    beta1 = (
        c[1] / c[0]
    )

    beta2 = (
        c[3] / c[2]
    )

    equality_residual = (
        c[1] * c[2]
        - c[0] * c[3]
    )

    if (
        beta1 < beta_lower - 1e-10
        or beta1 > beta_upper + 1e-10
    ):
        raise RuntimeError(
            "Recovered beta1 violates bounds."
        )

    if (
        beta2 < beta_lower - 1e-10
        or beta2 > beta_upper + 1e-10
    ):
        raise RuntimeError(
            "Recovered beta2 violates bounds."
        )

    equality_scale = max(
        abs(c[1] * c[2]),
        abs(c[0] * c[3]),
        1.0
    )

    if (
        abs(equality_residual)
        > 1e-10 * equality_scale
    ):
        raise RuntimeError(
            "Equality constraint was not "
            "satisfied numerically."
        )

    return c
# ----- Test 2---
# c_est = solve_linear_coefficients(Phi, d_clean)
# print(c_est)
# -------

def varpro_objective(theta, TI, TE, d):
    """
    Compute the reduced VarPro Objective
        F(theta) = 0.5 * ||Phi(theta) c(theta)- d||_2^2
    
    where c(theta) is obtained from the linear least squares.

    """

    Phi = build_design_matrix(TI, TE, theta)

    c = solve_linear_coefficients(Phi, d)

    residual = Phi @ c - d

    objective = 0.5 * np.dot(residual, residual)

    return objective

# ---- Test 3---
# F_true = varpro_objective(
#     theta_true,
#     TI,
#     TE,
#     d_clean 
# )

# F0 = varpro_objective(
#     THETA0,
#     TI,
#     TE,
#     d_clean 
# )

# print(F_true)
# print(F0)

def varpro_gradient(theta, TI, TE, d):
    """
    Compute the gradient of the reduced VarPro Objective

         F(theta) = 0.5 * ||Phi(theta)c*(theta) - d||_2^2
    using 
        dF/dtheta_j
        =
        r^T (dPhi/dtheta_j) c*
    
    where c* is obtained from the constrained linear SLSQP problem.
    """
    Phi = build_design_matrix(TI, TE, theta)

    c = solve_linear_coefficients(Phi, d)

    residual = Phi @ c - d

    dPhi_dT11, dPhi_dT12, dPhi_dT21, dPhi_dT22 = (
        build_design_matrix_derivatives(
            TI,
            TE,
            theta
        )
    )

    # dF/d theta_i
    dF_dT11 = residual @ (dPhi_dT11 @ c)
    dF_dT12 = residual @ (dPhi_dT12 @ c)
    dF_dT21 = residual @ (dPhi_dT21 @ c)
    dF_dT22 = residual @ (dPhi_dT22 @ c)

    gradient = np.array([
        dF_dT11,
        dF_dT12,
        dF_dT21,
        dF_dT22
    ])

    return gradient
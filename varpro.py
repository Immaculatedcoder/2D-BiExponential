import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

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

def solve_linear_coefficients(Phi, d):
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
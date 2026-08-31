"""
varpro_equal_beta.py

Equal-beta Variable Projection for the 2-D biexponential model.

This module enforces

    beta1 = beta2 = beta

by reparameterization rather than by imposing the nonlinear equality

    k1_hat*k2 - k1*k2_hat = 0

inside an SLSQP solve.

Model
-----
For alpha = [T11, T12, T21, T22, beta],

    S(TI, TE)
      = k1 * (1 - beta*exp(-TI/T11)) * exp(-TE/T21)
      + k2 * (1 - beta*exp(-TI/T12)) * exp(-TE/T22).

For fixed alpha, the model is linear in

    k = [k1, k2].

Thus

    d ~= Psi(alpha) k,

and VarPro eliminates k by solving a two-variable bounded linear
least-squares problem.

Important
---------
All vectorization uses order="F", matching the existing project.
"""

import numpy as np
from scipy.optimize import lsq_linear


# ============================================================
# Design matrix
# ============================================================

def build_design_matrix_equal_beta(TI, TE, alpha):
    """
    Build the equal-beta VarPro design matrix Psi(alpha).

    Parameters
    ----------
    TI : array_like
        Inversion-time grid, shape (n_TI,).

    TE : array_like
        Echo-time grid, shape (n_TE,).

    alpha : array_like, shape (5,)
        Nonlinear parameters

            alpha = [T11, T12, T21, T22, beta].

    Returns
    -------
    Psi : ndarray, shape (n_TI*n_TE, 2)

        Column 1:
            (1 - beta*exp(-TI/T11))*exp(-TE/T21)

        Column 2:
            (1 - beta*exp(-TI/T12))*exp(-TE/T22)
    """

    alpha = np.asarray(alpha, dtype=float)

    if alpha.shape != (5,):
        raise ValueError(
            "alpha must have shape (5,) with "
            "[T11, T12, T21, T22, beta]."
        )

    T11, T12, T21, T22, beta = alpha

    if min(T11, T12, T21, T22) <= 0.0:
        raise ValueError("All relaxation times must be positive.")

    TI_grid, TE_grid = np.meshgrid(
        np.asarray(TI, dtype=float),
        np.asarray(TE, dtype=float),
        indexing="ij"
    )

    E11 = np.exp(-TI_grid / T11)
    E12 = np.exp(-TI_grid / T12)
    E21 = np.exp(-TE_grid / T21)
    E22 = np.exp(-TE_grid / T22)

    psi1 = (1.0 - beta * E11) * E21
    psi2 = (1.0 - beta * E12) * E22

    Psi = np.column_stack([
        psi1.ravel(order="F"),
        psi2.ravel(order="F"),
    ])

    return Psi


# ============================================================
# Design-matrix derivatives
# ============================================================

def build_design_matrix_derivatives_equal_beta(TI, TE, alpha):
    """
    Build analytic derivatives of Psi(alpha).

    alpha = [T11, T12, T21, T22, beta]

    Returns
    -------
    dPsi_dT11,
    dPsi_dT12,
    dPsi_dT21,
    dPsi_dT22,
    dPsi_dbeta

    Each derivative has shape (n_TI*n_TE, 2).
    """

    alpha = np.asarray(alpha, dtype=float)

    if alpha.shape != (5,):
        raise ValueError(
            "alpha must have shape (5,) with "
            "[T11, T12, T21, T22, beta]."
        )

    T11, T12, T21, T22, beta = alpha

    if min(T11, T12, T21, T22) <= 0.0:
        raise ValueError("All relaxation times must be positive.")

    TI_grid, TE_grid = np.meshgrid(
        np.asarray(TI, dtype=float),
        np.asarray(TE, dtype=float),
        indexing="ij"
    )

    E11 = np.exp(-TI_grid / T11)
    E12 = np.exp(-TI_grid / T12)
    E21 = np.exp(-TE_grid / T21)
    E22 = np.exp(-TE_grid / T22)

    zero = np.zeros_like(TI_grid)

    # --------------------------------------------------------
    # dPsi / dT11
    #
    # psi1 = (1 - beta*E11)*E21
    # dE11/dT11 = (TI/T11^2)*E11
    # --------------------------------------------------------

    dpsi1_dT11 = (
        -beta
        * (TI_grid / T11**2)
        * E11
        * E21
    )

    dPsi_dT11 = np.column_stack([
        dpsi1_dT11.ravel(order="F"),
        zero.ravel(order="F"),
    ])

    # --------------------------------------------------------
    # dPsi / dT12
    # --------------------------------------------------------

    dpsi2_dT12 = (
        -beta
        * (TI_grid / T12**2)
        * E12
        * E22
    )

    dPsi_dT12 = np.column_stack([
        zero.ravel(order="F"),
        dpsi2_dT12.ravel(order="F"),
    ])

    # --------------------------------------------------------
    # dPsi / dT21
    # --------------------------------------------------------

    dpsi1_dT21 = (
        (1.0 - beta * E11)
        * (TE_grid / T21**2)
        * E21
    )

    dPsi_dT21 = np.column_stack([
        dpsi1_dT21.ravel(order="F"),
        zero.ravel(order="F"),
    ])

    # --------------------------------------------------------
    # dPsi / dT22
    # --------------------------------------------------------

    dpsi2_dT22 = (
        (1.0 - beta * E12)
        * (TE_grid / T22**2)
        * E22
    )

    dPsi_dT22 = np.column_stack([
        zero.ravel(order="F"),
        dpsi2_dT22.ravel(order="F"),
    ])

    # --------------------------------------------------------
    # dPsi / dbeta
    # --------------------------------------------------------

    dpsi1_dbeta = -E11 * E21
    dpsi2_dbeta = -E12 * E22

    dPsi_dbeta = np.column_stack([
        dpsi1_dbeta.ravel(order="F"),
        dpsi2_dbeta.ravel(order="F"),
    ])

    return (
        dPsi_dT11,
        dPsi_dT12,
        dPsi_dT21,
        dPsi_dT22,
        dPsi_dbeta,
    )


# ============================================================
# Inner bounded linear least-squares solve
# ============================================================

def solve_linear_coefficients_equal_beta(
    Psi,
    d,
    eps_k=1e-8
):
    """
    Solve the linear VarPro subproblem

        min_k  0.5 * ||Psi k - d||_2^2

    subject to

        k1 >= eps_k,
        k2 >= eps_k.

    Because beta is now an explicit nonlinear parameter, there are
    only two linear coefficients:

        k = [k1, k2].

    Parameters
    ----------
    Psi : ndarray, shape (m, 2)

    d : ndarray, shape (m,)

    eps_k : float
        Strictly positive lower bound used numerically for k1 and k2.

    Returns
    -------
    k : ndarray, shape (2,)
    """

    Psi = np.asarray(Psi, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    if Psi.ndim != 2 or Psi.shape[1] != 2:
        raise ValueError("Psi must have shape (m, 2).")

    if Psi.shape[0] != d.size:
        raise ValueError(
            "Psi and d have incompatible dimensions."
        )

    if eps_k <= 0.0:
        raise ValueError("eps_k must be positive.")

    result = lsq_linear(
        Psi,
        d,
        bounds=(
            np.array([eps_k, eps_k], dtype=float),
            np.array([np.inf, np.inf], dtype=float)
        ),
        method="trf",
        lsmr_tol="auto",
        verbose=0
    )

    if not result.success:
        raise RuntimeError(
            "Equal-beta linear least-squares solve failed: "
            f"{result.message}"
        )

    return result.x


# ============================================================
# Reduced VarPro objective
# ============================================================

def varpro_objective_equal_beta(
    alpha,
    TI,
    TE,
    d,
    eps_k=1e-8
):
    """
    Reduced equal-beta VarPro objective

        F(alpha)
          = 0.5 * ||Psi(alpha) k*(alpha) - d||_2^2,

    where

        alpha = [T11, T12, T21, T22, beta]

    and k*(alpha) is obtained from the bounded linear least-squares
    subproblem.
    """

    Psi = build_design_matrix_equal_beta(
        TI,
        TE,
        alpha
    )

    k = solve_linear_coefficients_equal_beta(
        Psi,
        d,
        eps_k=eps_k
    )

    residual = Psi @ k - d

    return 0.5 * float(
        residual @ residual
    )


# ============================================================
# Reduced VarPro gradient
# ============================================================

def varpro_gradient_equal_beta(
    alpha,
    TI,
    TE,
    d,
    eps_k=1e-8
):
    """
    Analytic reduced gradient using the envelope theorem.

    Since the inner constraints on k are independent of alpha,

        dF/dalpha_j
          = r^T (dPsi/dalpha_j) k*,

    where

        r = Psi k* - d.

    The returned ordering is

        [dF/dT11,
         dF/dT12,
         dF/dT21,
         dF/dT22,
         dF/dbeta].
    """

    Psi = build_design_matrix_equal_beta(
        TI,
        TE,
        alpha
    )

    k = solve_linear_coefficients_equal_beta(
        Psi,
        d,
        eps_k=eps_k
    )

    residual = Psi @ k - d

    derivatives = (
        build_design_matrix_derivatives_equal_beta(
            TI,
            TE,
            alpha
        )
    )

    gradient = np.array(
        [
            residual @ (dPsi @ k)
            for dPsi in derivatives
        ],
        dtype=float
    )

    return gradient


# ============================================================
# Optional diagnostic: finite-difference gradient check
# ============================================================

def check_gradient_equal_beta(
    alpha,
    TI,
    TE,
    d,
    eps_k=1e-8,
    relative_step=1e-6
):
    """
    Compare the analytic reduced gradient with centered finite
    differences.

    This is intended as a debugging / validation helper.

    Returns
    -------
    diagnostics : dict
        Contains analytic, finite_difference, absolute_error,
        and relative_error arrays.
    """

    alpha = np.asarray(alpha, dtype=float)

    analytic = varpro_gradient_equal_beta(
        alpha,
        TI,
        TE,
        d,
        eps_k=eps_k
    )

    finite_difference = np.zeros(5, dtype=float)

    for j in range(5):

        h = relative_step * max(
            1.0,
            abs(alpha[j])
        )

        alpha_plus = alpha.copy()
        alpha_minus = alpha.copy()

        alpha_plus[j] += h
        alpha_minus[j] -= h

        f_plus = varpro_objective_equal_beta(
            alpha_plus,
            TI,
            TE,
            d,
            eps_k=eps_k
        )

        f_minus = varpro_objective_equal_beta(
            alpha_minus,
            TI,
            TE,
            d,
            eps_k=eps_k
        )

        finite_difference[j] = (
            (f_plus - f_minus)
            / (2.0 * h)
        )

    absolute_error = np.abs(
        analytic - finite_difference
    )

    scale = np.maximum(
        1.0,
        np.maximum(
            np.abs(analytic),
            np.abs(finite_difference)
        )
    )

    relative_error = (
        absolute_error / scale
    )

    return {
        "analytic": analytic,
        "finite_difference": finite_difference,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


# ============================================================
# Convenient aliases
# ============================================================

build_design_matrix = build_design_matrix_equal_beta

build_design_matrix_derivatives = (
    build_design_matrix_derivatives_equal_beta
)

solve_linear_coefficients = (
    solve_linear_coefficients_equal_beta
)

varpro_objective = (
    varpro_objective_equal_beta
)

varpro_gradient = (
    varpro_gradient_equal_beta
)

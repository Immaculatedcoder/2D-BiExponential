"""
estimate_equal_beta.py

High-level parameter-estimation wrapper for equal-beta VarPro.

This file mirrors the role of the existing estimate.py but enforces

    beta1 = beta2 = beta

by construction.

Returned physical parameter dictionary
--------------------------------------

    {
        "k1": ...,
        "k2": ...,
        "beta1": beta,
        "beta2": beta,
        "T11": ...,
        "T12": ...,
        "T21": ...,
        "T22": ...
    }

The function returns

    estimate, result

just like the existing SciPy VarPro pipeline.
"""

import numpy as np

from optimizer_equal_beta import (
    solve_theta_equal_beta,
)

from varpro_equal_beta import (
    build_design_matrix_equal_beta,
    solve_linear_coefficients_equal_beta,
)


def estimate_parameters_equal_beta(
    D,
    TI,
    TE,
    theta0,
    theta_lower,
    theta_upper,
    beta0=1.0,
    beta_lower=0.0,
    beta_upper=2.0,
    eps_k=1e-8,
    options=None
):
    """
    Estimate the physical parameters under the equality

        beta1 = beta2 = beta.

    Parameters
    ----------
    D : ndarray, shape (len(TI), len(TE))
        Two-dimensional data array.

    TI, TE : array_like
        Sampling grids.

    theta0 : ndarray, shape (4,)
        Initial relaxation-time vector

            [T11, T12, T21, T22].

    theta_lower, theta_upper : ndarray, shape (4,)
        Relaxation-time bounds.

    beta0 : float, default=1.0
        Initial common beta for the outer optimizer.

    beta_lower, beta_upper : float
        Bounds on the common beta.

    eps_k : float
        Positive lower bound on k1 and k2.

    options : dict or None
        Optional L-BFGS-B options.

    Returns
    -------
    estimate : dict
        Physical estimates with keys

            k1, k2, beta1, beta2,
            T11, T12, T21, T22.

    result : scipy.optimize.OptimizeResult
        Outer equal-beta L-BFGS-B result.

        result.x is ordered as

            [T11, T12, T21, T22, beta].
    """

    D = np.asarray(
        D,
        dtype=float
    )

    expected_shape = (
        len(TI),
        len(TE)
    )

    if D.shape != expected_shape:
        raise ValueError(
            f"D must have shape {expected_shape}, "
            f"but received {D.shape}."
        )

    # --------------------------------------------------------
    # Step 1: vectorize exactly as in the existing project
    # --------------------------------------------------------

    d = D.ravel(
        order="F"
    )

    # --------------------------------------------------------
    # Step 2: solve the nonlinear reduced problem
    #
    # alpha = [T11, T12, T21, T22, beta]
    # --------------------------------------------------------

    result = solve_theta_equal_beta(
        theta0,
        TI,
        TE,
        d,
        theta_lower,
        theta_upper,
        beta0=beta0,
        beta_lower=beta_lower,
        beta_upper=beta_upper,
        eps_k=eps_k,
        options=options
    )

    if not result.success:
        raise RuntimeError(
            "Equal-beta outer VarPro solve failed: "
            f"{result.message}"
        )

    alpha_est = np.asarray(
        result.x,
        dtype=float
    )

    T11, T12, T21, T22, beta = (
        alpha_est
    )

    # --------------------------------------------------------
    # Step 3: recover the optimal linear coefficients
    #
    # k = [k1, k2]
    # --------------------------------------------------------

    Psi_est = (
        build_design_matrix_equal_beta(
            TI,
            TE,
            alpha_est
        )
    )

    k_est = (
        solve_linear_coefficients_equal_beta(
            Psi_est,
            d,
            eps_k=eps_k
        )
    )

    k1 = float(
        k_est[0]
    )

    k2 = float(
        k_est[1]
    )

    beta = float(
        beta
    )

    # --------------------------------------------------------
    # Step 4: bundle physical parameters
    #
    # Equality is exact by construction.
    # --------------------------------------------------------

    estimate = {
        "k1": k1,
        "k2": k2,
        "beta1": beta,
        "beta2": beta,
        "T11": float(T11),
        "T12": float(T12),
        "T21": float(T21),
        "T22": float(T22),
    }

    # --------------------------------------------------------
    # Step 5: add useful diagnostics to OptimizeResult
    # --------------------------------------------------------

    residual = (
        Psi_est @ k_est
        - d
    )

    result["theta_relaxation"] = (
        alpha_est[:4].copy()
    )

    result["beta"] = beta

    result["k"] = (
        k_est.copy()
    )

    result["residual_norm"] = float(
        np.linalg.norm(
            residual
        )
    )

    result["objective_recomputed"] = float(
        0.5
        * residual
        @ residual
    )

    return estimate, result


# ============================================================
# Drop-in-style alias
#
# This permits
#
#     from estimate_equal_beta import estimate_parameters
#
# if desired.
# ============================================================

estimate_parameters = (
    estimate_parameters_equal_beta
)

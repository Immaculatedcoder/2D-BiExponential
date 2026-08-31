"""
optimizer_equal_beta.py

Outer nonlinear optimizer for the equal-beta VarPro formulation.

The nonlinear vector is

    alpha = [T11, T12, T21, T22, beta].

The relaxation-time bounds are taken from the existing project, while

    beta_lower <= beta <= beta_upper

is imposed directly by L-BFGS-B.

The default beta interval is [0, 2].
"""

import numpy as np
from scipy.optimize import minimize, Bounds

from varpro_equal_beta import (
    varpro_objective_equal_beta,
    varpro_gradient_equal_beta,
)


def build_equal_beta_initial_vector(
    theta0,
    beta0=1.0
):
    """
    Convert the existing four-parameter initial theta into

        alpha0 = [T11, T12, T21, T22, beta0].
    """

    theta0 = np.asarray(
        theta0,
        dtype=float
    )

    if theta0.shape != (4,):
        raise ValueError(
            "theta0 must have shape (4,) with "
            "[T11, T12, T21, T22]."
        )

    return np.concatenate(
        [
            theta0,
            np.array([float(beta0)])
        ]
    )


def build_equal_beta_bounds(
    theta_lower,
    theta_upper,
    beta_lower=0.0,
    beta_upper=2.0
):
    """
    Build bounds for

        alpha = [T11, T12, T21, T22, beta].
    """

    theta_lower = np.asarray(
        theta_lower,
        dtype=float
    )

    theta_upper = np.asarray(
        theta_upper,
        dtype=float
    )

    if theta_lower.shape != (4,):
        raise ValueError(
            "theta_lower must have shape (4,)."
        )

    if theta_upper.shape != (4,):
        raise ValueError(
            "theta_upper must have shape (4,)."
        )

    if np.any(theta_lower >= theta_upper):
        raise ValueError(
            "Each theta lower bound must be smaller "
            "than its upper bound."
        )

    if beta_lower >= beta_upper:
        raise ValueError(
            "beta_lower must be smaller than beta_upper."
        )

    lower = np.concatenate(
        [
            theta_lower,
            np.array([float(beta_lower)])
        ]
    )

    upper = np.concatenate(
        [
            theta_upper,
            np.array([float(beta_upper)])
        ]
    )

    return Bounds(
        lb=lower,
        ub=upper
    )


def solve_theta_equal_beta(
    theta0,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    beta0=1.0,
    beta_lower=0.0,
    beta_upper=2.0,
    eps_k=1e-8,
    options=None
):
    """
    Solve the equal-beta reduced VarPro problem

        min_alpha F(alpha)

    subject to

        theta_lower <= [T11,T12,T21,T22] <= theta_upper
        beta_lower  <= beta <= beta_upper.

    Parameters
    ----------
    theta0 : ndarray, shape (4,)
        Existing relaxation-time initial guess.

    TI, TE : array_like
        Sampling grids.

    d : ndarray, shape (m,)
        Vectorized data, using order="F".

    theta_lower, theta_upper : ndarray, shape (4,)
        Relaxation-time bounds.

    beta0 : float, default=1.0
        Initial common beta.

        It is deliberately not set to the ground-truth value by
        default. The midpoint 1.0 of [0,2] is used as a neutral
        starting value.

    beta_lower, beta_upper : float
        Common-beta bounds.

    eps_k : float
        Positive lower bound for k1 and k2 in the inner solve.

    options : dict or None
        Optional scipy.optimize.minimize options.

    Returns
    -------
    result : scipy.optimize.OptimizeResult

        result.x is ordered as

            [T11, T12, T21, T22, beta].
    """

    alpha0 = build_equal_beta_initial_vector(
        theta0,
        beta0=beta0
    )

    bounds = build_equal_beta_bounds(
        theta_lower,
        theta_upper,
        beta_lower=beta_lower,
        beta_upper=beta_upper
    )

    # Clip the initial beta to the allowed interval if a user
    # supplied a slightly out-of-range value.
    alpha0 = np.minimum(
        np.maximum(
            alpha0,
            bounds.lb
        ),
        bounds.ub
    )

    def objective(alpha):
        return varpro_objective_equal_beta(
            alpha,
            TI,
            TE,
            d,
            eps_k=eps_k
        )

    def gradient(alpha):
        return varpro_gradient_equal_beta(
            alpha,
            TI,
            TE,
            d,
            eps_k=eps_k
        )

    result = minimize(
        objective,
        alpha0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options=options
    )

    return result


# Convenient alias
solve_theta = solve_theta_equal_beta

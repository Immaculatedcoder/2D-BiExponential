import numpy as np

from solver_scipy import solve_scipy


def estimate_parameters_scipy(
    D,
    TI,
    TE,
    theta0,
    theta_lower,
    theta_upper,
    beta0=1.0
):
    """
    Estimate physical biexponential parameters
    using direct SciPy SLSQP optimization.

    Parameters
    ----------
    D : ndarray
        2-D measured data.

    TI, TE : ndarray
        Sampling grids.

    theta0 : ndarray
        Initial nonlinear guess:

            [T11, T12, T21, T22]

    theta_lower : ndarray
        Lower bounds for theta.

    theta_upper : ndarray
        Upper bounds for theta.

    beta0 : float
        Common beta used only to build
        the initial feasible coefficient vector.

    Returns
    -------
    estimate : dict
        Physical parameter estimates.

    diagnostics : dict
        Solver diagnostics.

    result : OptimizeResult
        Raw SciPy result.
    """

    # ----------------------------------
    # Vectorize the 2-D data
    # ----------------------------------

    d = np.asarray(
        D,
        dtype=float
    ).ravel(
        order="F"
    )

    # ----------------------------------
    # Direct SLSQP solve
    # ----------------------------------

    (
        c_star,
        theta_star,
        beta1,
        beta2,
        diagnostics,
        result
    ) = solve_scipy(
        TI,
        TE,
        d,
        theta0,
        theta_lower,
        theta_upper,
        beta0=beta0
    )

    # ----------------------------------
    # Recover physical parameters
    # ----------------------------------

    k1 = c_star[0]
    k2 = c_star[2]

    T11 = theta_star[0]
    T12 = theta_star[1]
    T21 = theta_star[2]
    T22 = theta_star[3]

    estimate = {
        "k1": k1,
        "k2": k2,
        "beta1": beta1,
        "beta2": beta2,
        "T11": T11,
        "T12": T12,
        "T21": T21,
        "T22": T22
    }

    return (
        estimate,
        diagnostics,
        result
    )
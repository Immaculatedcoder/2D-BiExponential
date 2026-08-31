import numpy as np
from scipy.integrate import solve_ivp

from varpro import (
    build_design_matrix,
    build_design_matrix_derivatives,
)


# ============================================================
# Equality constraint
#
# h(c) = 0.5 * c.T @ H @ c
#
# c = [k1, k1_hat, k2, k2_hat]
#
# h(c) = k1_hat*k2 - k1*k2_hat
# ============================================================

H = np.array([
    [ 0.0,  0.0,  0.0, -1.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0,  0.0],
], dtype=float)


# ============================================================
# Inequality constraints
#
# A.T @ c + q <= 0
#
# Constraints:
#
# k1 >= eps_k
# k1_hat >= 0
# k1_hat <= 2*k1
#
# k2 >= eps_k
# k2_hat >= 0
# k2_hat <= 2*k2
# ============================================================

A = np.array([
    [-1.0,  0.0, -2.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  1.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0, -1.0,  0.0, -2.0],
    [ 0.0,  0.0,  0.0,  0.0, -1.0,  1.0],
], dtype=float)


def build_q(eps_k=1e-8):
    """
    Build q for

        A.T @ c + q <= 0.
    """

    return np.array([
        eps_k,
        0.0,
        0.0,
        eps_k,
        0.0,
        0.0,
    ], dtype=float)


# ============================================================
# Reduced linear problem
# ============================================================

def build_B(TI, TE, theta, lam):
    """
    B(theta, lambda)
        = Phi(theta).T @ Phi(theta) + lambda * H
    """

    Phi = build_design_matrix(
        TI,
        TE,
        theta
    )

    B = Phi.T @ Phi + lam * H

    return B


def solve_reduced_c(
    TI,
    TE,
    theta,
    mu,
    lam,
    d
):
    """
    Recover

        c* = argmin_c L(c, theta, mu, lambda)

    by solving

        B c* = Phi.T d - A mu.
    """

    Phi = build_design_matrix(
        TI,
        TE,
        theta
    )

    B = Phi.T @ Phi + lam * H

    rhs = Phi.T @ d - A @ mu

    # Never form B^{-1} explicitly
    c = np.linalg.solve(
        B,
        rhs
    )

    return c


# ============================================================
# Reduced quantities
# ============================================================

def reduced_quantities(
    TI,
    TE,
    theta,
    mu,
    lam,
    d,
    eps_k=1e-8
):
    """
    Returns

        c : reduced linear coefficients
        r : data residual
        g : inequality residual
        h : equality residual
    """

    c = solve_reduced_c(
        TI,
        TE,
        theta,
        mu,
        lam,
        d
    )

    Phi = build_design_matrix(
        TI,
        TE,
        theta
    )

    r = Phi @ c - d

    q = build_q(
        eps_k
    )

    g = A.T @ c + q

    h = 0.5 * c @ H @ c

    return c, r, g, h


# ============================================================
# Reduced gradients
# ============================================================

def reduced_gradients(
    TI,
    TE,
    theta,
    mu,
    lam,
    d,
    eps_k=1e-8
):
    """
    Reduced Lagrangian gradients

        dL_bar/dtheta_j = r.T (dPhi_j c)

        grad_mu = g

        grad_lambda = h
    """

    # IMPORTANT FIX:
    # propagate eps_k supplied by caller
    c, r, g, h = reduced_quantities(
        TI,
        TE,
        theta,
        mu,
        lam,
        d,
        eps_k=eps_k
    )

    (
        dPhi_dT11,
        dPhi_dT12,
        dPhi_dT21,
        dPhi_dT22
    ) = build_design_matrix_derivatives(
        TI,
        TE,
        theta
    )

    grad_theta = np.array([
        r @ (dPhi_dT11 @ c),
        r @ (dPhi_dT12 @ c),
        r @ (dPhi_dT21 @ c),
        r @ (dPhi_dT22 @ c),
    ], dtype=float)

    grad_mu = g

    grad_lam = h

    return (
        grad_theta,
        grad_mu,
        grad_lam
    )


# ============================================================
# Projection operators
# ============================================================

def project_mu_direction(
    mu,
    direction,
    tol=1e-12
):
    """
    Project the dual ascent direction onto the
    tangent cone of

        mu >= 0.

    If mu_i is already at zero, it cannot move
    farther into the negative region.
    """

    mu = np.asarray(
        mu,
        dtype=float
    )

    projected = np.asarray(
        direction,
        dtype=float
    ).copy()

    for i in range(len(mu)):

        if (
            mu[i] <= tol
            and projected[i] < 0.0
        ):
            projected[i] = 0.0

    return projected


def project_theta_direction(
    theta,
    direction,
    theta_lower,
    theta_upper,
    tol=1e-12
):
    """
    Project the theta direction onto the
    tangent cone of the parameter box.
    """

    theta = np.asarray(
        theta,
        dtype=float
    )

    direction = np.asarray(
        direction,
        dtype=float
    )

    theta_lower = np.asarray(
        theta_lower,
        dtype=float
    )

    theta_upper = np.asarray(
        theta_upper,
        dtype=float
    )

    projected = direction.copy()

    for j in range(len(theta)):

        # At lower bound:
        # cannot move below lower bound
        if (
            theta[j] <= theta_lower[j] + tol
            and projected[j] < 0.0
        ):
            projected[j] = 0.0

        # At upper bound:
        # cannot move above upper bound
        elif (
            theta[j] >= theta_upper[j] - tol
            and projected[j] > 0.0
        ):
            projected[j] = 0.0

    return projected


# ============================================================
# Preconditioner-independent stationarity residual
# ============================================================

def projected_stationarity_residual(
    theta,
    mu,
    lam,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    eps_k=1e-8
):
    """
    Projected saddle/KKT stationarity residual.

    IMPORTANT:

    This residual contains NO gamma values and NO
    preconditioners.

    Therefore it gives us the same convergence
    criterion for both the ordinary and
    preconditioned saddle flows.

    theta block:
        projected(-grad_theta)

    mu block:
        projected(+grad_mu)

    lambda block:
        grad_lambda = h
    """

    (
        grad_theta,
        grad_mu,
        grad_lam
    ) = reduced_gradients(
        TI,
        TE,
        theta,
        mu,
        lam,
        d,
        eps_k=eps_k
    )

    theta_stationarity = (
        project_theta_direction(
            theta,
            -grad_theta,
            theta_lower,
            theta_upper
        )
    )

    mu_stationarity = (
        project_mu_direction(
            mu,
            grad_mu
        )
    )

    stationarity = np.concatenate([
        theta_stationarity,
        mu_stationarity,
        np.array(
            [grad_lam],
            dtype=float
        )
    ])

    return stationarity


def projected_stationarity_norm(
    theta,
    mu,
    lam,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    eps_k=1e-8
):
    """
    Euclidean norm of the projected
    stationarity residual.
    """

    residual = (
        projected_stationarity_residual(
            theta,
            mu,
            lam,
            TI,
            TE,
            d,
            theta_lower,
            theta_upper,
            eps_k=eps_k
        )
    )

    return np.linalg.norm(
        residual
    )


# ============================================================
# UNPRECONDITIONED saddle RHS
# ============================================================

def saddle_rhs(
    t,
    y,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    gamma_theta=1.0,
    gamma_mu=1.0,
    gamma_lam=1.0,
    eps_k=1e-8
):
    """
    Reduced projected Arrow-Hurwicz-Uzawa
    saddle dynamics.

    State

        y = [theta, mu, lambda]

    where

        theta : 4 variables
        mu    : 6 variables
        lambda: 1 variable
    """

    theta = y[0:4]
    mu = y[4:10]
    lam = y[10]

    (
        grad_theta,
        grad_mu,
        grad_lam
    ) = reduced_gradients(
        TI,
        TE,
        theta,
        mu,
        lam,
        d,

        # IMPORTANT FIX
        eps_k=eps_k
    )

    # --------------------------------------------------------
    # theta descent
    # --------------------------------------------------------

    theta_direction = (
        -gamma_theta
        * grad_theta
    )

    theta_dot = (
        project_theta_direction(
            theta,
            theta_direction,
            theta_lower,
            theta_upper
        )
    )

    # --------------------------------------------------------
    # mu ascent
    # --------------------------------------------------------

    mu_direction = (
        gamma_mu
        * grad_mu
    )

    mu_dot = (
        project_mu_direction(
            mu,
            mu_direction
        )
    )

    # --------------------------------------------------------
    # lambda ascent
    # --------------------------------------------------------

    lambda_dot = (
        gamma_lam
        * grad_lam
    )

    y_dot = np.concatenate([
        theta_dot,
        mu_dot,
        np.array(
            [lambda_dot],
            dtype=float
        )
    ])

    return y_dot


# ============================================================
# Input validation
# ============================================================

def _validate_solver_inputs(
    theta0,
    mu0,
    lam0,
    theta_lower,
    theta_upper
):
    """
    Validate initial saddle state.
    """

    theta0 = np.asarray(
        theta0,
        dtype=float
    )

    mu0 = np.asarray(
        mu0,
        dtype=float
    )

    theta_lower = np.asarray(
        theta_lower,
        dtype=float
    )

    theta_upper = np.asarray(
        theta_upper,
        dtype=float
    )

    if theta0.shape != (4,):
        raise ValueError(
            "theta0 must have shape (4,)."
        )

    if mu0.shape != (6,):
        raise ValueError(
            "mu0 must have shape (6,)."
        )

    if (
        theta_lower.shape != (4,)
        or theta_upper.shape != (4,)
    ):
        raise ValueError(
            "theta_lower and theta_upper "
            "must both have shape (4,)."
        )

    if np.any(
        theta_lower >= theta_upper
    ):
        raise ValueError(
            "Every theta lower bound must "
            "be smaller than its upper bound."
        )

    if (
        np.any(theta0 < theta_lower)
        or np.any(theta0 > theta_upper)
    ):
        raise ValueError(
            "Initial theta must lie "
            "inside the theta bounds."
        )

    if np.any(
        mu0 < 0.0
    ):
        raise ValueError(
            "Initial mu must satisfy "
            "mu >= 0."
        )

    if not np.isfinite(
        lam0
    ):
        raise ValueError(
            "lam0 must be finite."
        )

    return (
        theta0,
        mu0,
        theta_lower,
        theta_upper
    )


def _validate_initial_B(
    TI,
    TE,
    theta0,
    lam0,
    b_eig_tol
):
    """
    Require B(theta0, lambda0) to be
    positive definite.
    """

    B0 = build_B(
        TI,
        TE,
        theta0,
        lam0
    )

    min_eig_B0 = np.min(
        np.linalg.eigvalsh(B0)
    )

    if min_eig_B0 <= b_eig_tol:

        raise ValueError(
            "Initial B is not sufficiently "
            "positive definite. "
            f"min_eig(B0) = {min_eig_B0:.6e}"
        )


def _validate_gammas(
    gamma_theta,
    gamma_mu,
    gamma_lam
):
    """
    Saddle-flow scaling parameters must
    be nonnegative.
    """

    if gamma_theta < 0.0:
        raise ValueError(
            "gamma_theta must be nonnegative."
        )

    if gamma_mu < 0.0:
        raise ValueError(
            "gamma_mu must be nonnegative."
        )

    if gamma_lam < 0.0:
        raise ValueError(
            "gamma_lam must be nonnegative."
        )


# ============================================================
# UNPRECONDITIONED solver
# ============================================================

def solve_saddle(
    theta0,
    mu0,
    lam0,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    t_span,
    gamma_theta=1.0,
    gamma_mu=1.0,
    gamma_lam=1.0,
    eps_k=1e-8,
    method="BDF",
    rtol=1e-6,
    atol=1e-8,
    max_step=np.inf,
    b_eig_tol=1e-8,
    steady_tol=1e-5
):
    """
    Integrate the unpreconditioned
    reduced saddle dynamics.

    Convergence is determined using the
    preconditioner-independent projected
    stationarity residual.
    """

    (
        theta0,
        mu0,
        theta_lower,
        theta_upper
    ) = _validate_solver_inputs(
        theta0,
        mu0,
        lam0,
        theta_lower,
        theta_upper
    )

    _validate_gammas(
        gamma_theta,
        gamma_mu,
        gamma_lam
    )

    _validate_initial_B(
        TI,
        TE,
        theta0,
        lam0,
        b_eig_tol
    )

    y0 = np.concatenate([
        theta0,
        mu0,
        np.array(
            [lam0],
            dtype=float
        )
    ])

    # --------------------------------------------------------
    # ODE RHS
    # --------------------------------------------------------

    def rhs(t, y):

        return saddle_rhs(
            t,
            y,
            TI,
            TE,
            d,
            theta_lower,
            theta_upper,
            gamma_theta=gamma_theta,
            gamma_mu=gamma_mu,
            gamma_lam=gamma_lam,
            eps_k=eps_k
        )

    # --------------------------------------------------------
    # Positive-definiteness event
    # --------------------------------------------------------

    def b_positive_definite_event(
        t,
        y
    ):

        theta = y[0:4]
        lam = y[10]

        B = build_B(
            TI,
            TE,
            theta,
            lam
        )

        min_eig = np.min(
            np.linalg.eigvalsh(B)
        )

        return (
            min_eig
            - b_eig_tol
        )

    b_positive_definite_event.terminal = True
    b_positive_definite_event.direction = -1

    # --------------------------------------------------------
    # Stationarity event
    # --------------------------------------------------------

    def steady_state_event(
        t,
        y
    ):

        theta = y[0:4]
        mu = y[4:10]
        lam = y[10]

        norm_value = (
            projected_stationarity_norm(
                theta,
                mu,
                lam,
                TI,
                TE,
                d,
                theta_lower,
                theta_upper,
                eps_k=eps_k
            )
        )

        return (
            norm_value
            - steady_tol
        )

    steady_state_event.terminal = True
    steady_state_event.direction = -1

    solution = solve_ivp(
        rhs,
        t_span,
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=[
            b_positive_definite_event,
            steady_state_event
        ]
    )

    return solution


# ============================================================
# THETA PRECONDITIONER
# ============================================================

def build_theta_preconditioner(
    theta_lower,
    theta_upper
):
    """
    Build the normalized range-based
    theta preconditioner.

    Let

        s = theta_upper - theta_lower.

    The raw range-based matrix is

        diag(s^2).

    We divide by max(s^2) so that the
    largest diagonal entry is 1.

    For the current bounds this gives

        diag(
            0.44444444,
            1.0,
            0.00751111,
            0.02777778
        )
    """

    theta_lower = np.asarray(
        theta_lower,
        dtype=float
    )

    theta_upper = np.asarray(
        theta_upper,
        dtype=float
    )

    if (
        theta_lower.shape != (4,)
        or theta_upper.shape != (4,)
    ):
        raise ValueError(
            "theta_lower and theta_upper "
            "must both have shape (4,)."
        )

    theta_scale = (
        theta_upper
        - theta_lower
    )

    if np.any(
        theta_scale <= 0.0
    ):
        raise ValueError(
            "All theta ranges must "
            "be strictly positive."
        )

    weights = (
        theta_scale ** 2
    )

    weights = (
        weights
        / np.max(weights)
    )

    P_theta = np.diag(
        weights
    )

    return P_theta


def _validate_preconditioner(
    P,
    shape,
    name
):
    """
    Require a symmetric positive-definite
    preconditioner.
    """

    P = np.asarray(
        P,
        dtype=float
    )

    if P.shape != shape:

        raise ValueError(
            f"{name} must have "
            f"shape {shape}."
        )

    if not np.all(
        np.isfinite(P)
    ):

        raise ValueError(
            f"{name} contains "
            "non-finite values."
        )

    if not np.allclose(
        P,
        P.T,
        rtol=1e-10,
        atol=1e-12
    ):

        raise ValueError(
            f"{name} must be symmetric."
        )

    min_eig = np.min(
        np.linalg.eigvalsh(P)
    )

    if min_eig <= 0.0:

        raise ValueError(
            f"{name} must be "
            "positive definite."
        )

    return P


# ============================================================
# PRECONDITIONED saddle RHS
# ============================================================

def saddle_rhs_preconditioned(
    t,
    y,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    P_theta,
    P_mu=None,
    p_lam=1.0,
    gamma_theta=1.0,
    gamma_mu=1.0,
    gamma_lam=1.0,
    eps_k=1e-8
):
    """
    Preconditioned reduced projected
    Arrow-Hurwicz-Uzawa dynamics.

    theta_dot =
        projected(
            -gamma_theta
            * P_theta
            * grad_theta
        )

    mu_dot =
        projected(
            gamma_mu
            * P_mu
            * grad_mu
        )

    lambda_dot =
        gamma_lam
        * p_lam
        * grad_lam
    """

    theta = y[0:4]
    mu = y[4:10]
    lam = y[10]

    (
        grad_theta,
        grad_mu,
        grad_lam
    ) = reduced_gradients(
        TI,
        TE,
        theta,
        mu,
        lam,
        d,
        eps_k=eps_k
    )

    if P_mu is None:
        P_mu = np.eye(6)

    # --------------------------------------------------------
    # theta descent
    # --------------------------------------------------------

    theta_direction = (
        -gamma_theta
        * (P_theta @ grad_theta)
    )

    theta_dot = (
        project_theta_direction(
            theta,
            theta_direction,
            theta_lower,
            theta_upper
        )
    )

    # --------------------------------------------------------
    # mu ascent
    # --------------------------------------------------------

    mu_direction = (
        gamma_mu
        * (P_mu @ grad_mu)
    )

    mu_dot = (
        project_mu_direction(
            mu,
            mu_direction
        )
    )

    # --------------------------------------------------------
    # lambda ascent
    # --------------------------------------------------------

    lambda_dot = (
        gamma_lam
        * p_lam
        * grad_lam
    )

    y_dot = np.concatenate([
        theta_dot,
        mu_dot,
        np.array(
            [lambda_dot],
            dtype=float
        )
    ])

    return y_dot


# ============================================================
# PRECONDITIONED solver
# ============================================================

def solve_saddle_preconditioned(
    theta0,
    mu0,
    lam0,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    t_span,
    P_theta,
    P_mu=None,
    p_lam=1.0,
    gamma_theta=1.0,
    gamma_mu=1.0,
    gamma_lam=1.0,
    eps_k=1e-8,
    method="BDF",
    rtol=1e-6,
    atol=1e-8,
    max_step=np.inf,
    b_eig_tol=1e-8,
    steady_tol=1e-5
):
    """
    Integrate the preconditioned
    reduced saddle dynamics.

    IMPORTANT:

    The ODE itself is preconditioned.

    However, the stopping test uses the
    ORIGINAL projected stationarity
    residual.

    Therefore changing P_theta does not
    artificially change what we mean by
    "converged".
    """

    (
        theta0,
        mu0,
        theta_lower,
        theta_upper
    ) = _validate_solver_inputs(
        theta0,
        mu0,
        lam0,
        theta_lower,
        theta_upper
    )

    _validate_gammas(
        gamma_theta,
        gamma_mu,
        gamma_lam
    )

    P_theta = (
        _validate_preconditioner(
            P_theta,
            (4, 4),
            "P_theta"
        )
    )

    if P_mu is None:
        P_mu = np.eye(6)

    P_mu = (
        _validate_preconditioner(
            P_mu,
            (6, 6),
            "P_mu"
        )
    )

    if (
        not np.isfinite(p_lam)
        or p_lam <= 0.0
    ):
        raise ValueError(
            "p_lam must be finite "
            "and strictly positive."
        )

    _validate_initial_B(
        TI,
        TE,
        theta0,
        lam0,
        b_eig_tol
    )

    y0 = np.concatenate([
        theta0,
        mu0,
        np.array(
            [lam0],
            dtype=float
        )
    ])

    # --------------------------------------------------------
    # PRECONDITIONED RHS
    # --------------------------------------------------------

    def rhs(t, y):

        return (
            saddle_rhs_preconditioned(
                t,
                y,
                TI,
                TE,
                d,
                theta_lower,
                theta_upper,
                P_theta,
                P_mu=P_mu,
                p_lam=p_lam,
                gamma_theta=gamma_theta,
                gamma_mu=gamma_mu,
                gamma_lam=gamma_lam,
                eps_k=eps_k
            )
        )

    # --------------------------------------------------------
    # B positive-definiteness event
    # --------------------------------------------------------

    def b_positive_definite_event(
        t,
        y
    ):

        theta = y[0:4]
        lam = y[10]

        B = build_B(
            TI,
            TE,
            theta,
            lam
        )

        min_eig = np.min(
            np.linalg.eigvalsh(B)
        )

        return (
            min_eig
            - b_eig_tol
        )

    b_positive_definite_event.terminal = True
    b_positive_definite_event.direction = -1

    # --------------------------------------------------------
    # PRECONDITIONER-INDEPENDENT convergence event
    # --------------------------------------------------------

    def steady_state_event(
        t,
        y
    ):

        theta = y[0:4]
        mu = y[4:10]
        lam = y[10]

        norm_value = (
            projected_stationarity_norm(
                theta,
                mu,
                lam,
                TI,
                TE,
                d,
                theta_lower,
                theta_upper,
                eps_k=eps_k
            )
        )

        return (
            norm_value
            - steady_tol
        )

    steady_state_event.terminal = True
    steady_state_event.direction = -1

    solution = solve_ivp(
        rhs,
        t_span,
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=[
            b_positive_definite_event,
            steady_state_event
        ]
    )

    return solution


# ============================================================
# Diagnostics
# ============================================================

def saddle_diagnostics(
    solution,
    TI,
    TE,
    d,
    theta_lower,
    theta_upper,
    eps_k=1e-8,
    preconditioned=False,
    P_theta=None,
    P_mu=None,
    p_lam=1.0,
    gamma_theta=1.0,
    gamma_mu=1.0,
    gamma_lam=1.0
):
    """
    Produce consistent diagnostics.

    We report TWO different quantities:

    1. flow_rhs
       The actual ODE velocity.

    2. stationarity_residual
       The original preconditioner-independent
       convergence measure.

    This prevents us from accidentally
    diagnosing a preconditioned solution
    with saddle_rhs().
    """

    y_final = (
        solution.y[:, -1]
    )

    theta_final = (
        y_final[0:4]
    )

    mu_final = (
        y_final[4:10]
    )

    lam_final = (
        y_final[10]
    )

    (
        c_final,
        r_final,
        g_final,
        h_final
    ) = reduced_quantities(
        TI,
        TE,
        theta_final,
        mu_final,
        lam_final,
        d,
        eps_k=eps_k
    )

    (
        grad_theta,
        grad_mu,
        grad_lam
    ) = reduced_gradients(
        TI,
        TE,
        theta_final,
        mu_final,
        lam_final,
        d,
        eps_k=eps_k
    )

    # --------------------------------------------------------
    # Actual flow RHS
    # --------------------------------------------------------

    if preconditioned:

        if P_theta is None:

            raise ValueError(
                "P_theta is required when "
                "preconditioned=True."
            )

        if P_mu is None:
            P_mu = np.eye(6)

        flow_rhs = (
            saddle_rhs_preconditioned(
                solution.t[-1],
                y_final,
                TI,
                TE,
                d,
                theta_lower,
                theta_upper,
                P_theta,
                P_mu=P_mu,
                p_lam=p_lam,
                gamma_theta=gamma_theta,
                gamma_mu=gamma_mu,
                gamma_lam=gamma_lam,
                eps_k=eps_k
            )
        )

    else:

        flow_rhs = saddle_rhs(
            solution.t[-1],
            y_final,
            TI,
            TE,
            d,
            theta_lower,
            theta_upper,
            gamma_theta=gamma_theta,
            gamma_mu=gamma_mu,
            gamma_lam=gamma_lam,
            eps_k=eps_k
        )

    # --------------------------------------------------------
    # Preconditioner-independent convergence residual
    # --------------------------------------------------------

    stationarity_residual = (
        projected_stationarity_residual(
            theta_final,
            mu_final,
            lam_final,
            TI,
            TE,
            d,
            theta_lower,
            theta_upper,
            eps_k=eps_k
        )
    )

    # --------------------------------------------------------
    # Final B
    # --------------------------------------------------------

    B_final = build_B(
        TI,
        TE,
        theta_final,
        lam_final
    )

    diagnostics = {

        "y":
            y_final,

        "theta":
            theta_final,

        "mu":
            mu_final,

        "lambda":
            lam_final,

        "c":
            c_final,

        "beta1":
            c_final[1]
            / c_final[0],

        "beta2":
            c_final[3]
            / c_final[2],

        "residual":
            r_final,

        "residual_norm":
            np.linalg.norm(
                r_final
            ),

        "g":
            g_final,

        "h":
            h_final,

        "grad_theta":
            grad_theta,

        "grad_mu":
            grad_mu,

        "grad_lam":
            grad_lam,

        "flow_rhs":
            flow_rhs,

        "flow_rhs_norm":
            np.linalg.norm(
                flow_rhs
            ),

        "stationarity_residual":
            stationarity_residual,

        "stationarity_norm":
            np.linalg.norm(
                stationarity_residual
            ),

        "B":
            B_final,

        "B_min_eigenvalue":
            np.min(
                np.linalg.eigvalsh(
                    B_final
                )
            ),

        "B_condition_number":
            np.linalg.cond(
                B_final
            )
    }

    return diagnostics


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from config import (
        TI,
        TE,
        TRUE_PARAMS,
        THETA_LOWER,
        THETA_UPPER,
        THETA0
    )

    from model import (
        biexponential_2d
    )

    # --------------------------------------------------------
    # Clean synthetic data
    # --------------------------------------------------------

    D_clean = biexponential_2d(
        TI,
        TE,
        TRUE_PARAMS
    )

    d_clean = (
        D_clean.ravel(
            order="F"
        )
    )

    theta_true = np.array([
        TRUE_PARAMS["T11"],
        TRUE_PARAMS["T12"],
        TRUE_PARAMS["T21"],
        TRUE_PARAMS["T22"],
    ], dtype=float)


    # ========================================================
    # Sanity check at exact solution
    # ========================================================

    (
        c_true,
        r_true,
        g_true,
        h_true
    ) = reduced_quantities(
        TI,
        TE,
        theta_true,
        np.zeros(6),
        0.0,
        d_clean
    )

    print(
        "\n================================"
    )
    print(
        "SANITY CHECK AT TRUE PARAMETERS"
    )
    print(
        "================================"
    )

    print("\nRecovered c:")
    print(c_true)

    print("\nResidual norm:")
    print(
        np.linalg.norm(
            r_true
        )
    )

    print("\nInequality residuals:")
    print(g_true)

    print("\nEquality residual h:")
    print(h_true)


    # ========================================================
    # Build theta preconditioner
    # ========================================================

    P_theta = (
        build_theta_preconditioner(
            THETA_LOWER,
            THETA_UPPER
        )
    )

    P_mu = np.eye(6)

    p_lam = 1.0

    print(
        "\n================================"
    )
    print(
        "THETA PRECONDITIONER"
    )
    print(
        "================================"
    )

    print(P_theta)


    # ========================================================
    # PRECONDITIONED RUN
    # ========================================================
    #
    # NOTE:
    #
    # The previous run stopped using the
    # PRECONDITIONED velocity norm.
    #
    # This corrected version uses the ORIGINAL
    # stationarity residual.
    #
    # Therefore give it more artificial time.
    # ========================================================

    solution = (
        solve_saddle_preconditioned(

            THETA0,

            np.zeros(6),

            0.0,

            TI,
            TE,

            d_clean,

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
    )


    # ========================================================
    # Diagnostics
    # ========================================================

    diagnostics = (
        saddle_diagnostics(

            solution,

            TI,
            TE,

            d_clean,

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
    )


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
        "Initial time:",
        solution.t[0]
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

    print("\nB-event times:")
    print(
        solution.t_events[0]
    )

    print(
        "\nSteady-state event times:"
    )
    print(
        solution.t_events[1]
    )


    print(
        "\n================================"
    )
    print(
        "FINAL ESTIMATES"
    )
    print(
        "================================"
    )

    print("\nFinal theta:")
    print(
        diagnostics["theta"]
    )

    print("\nFinal c:")
    print(
        diagnostics["c"]
    )

    print("\nFinal mu:")
    print(
        diagnostics["mu"]
    )

    print("\nFinal lambda:")
    print(
        diagnostics["lambda"]
    )

    print("\nFinal beta:")

    print(
        "beta1 =",
        diagnostics["beta1"]
    )

    print(
        "beta2 =",
        diagnostics["beta2"]
    )

    print("\nResidual norm:")
    print(
        diagnostics[
            "residual_norm"
        ]
    )

    print(
        "\nInequality residuals:"
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

    print(
        "\nFinal grad_theta:"
    )

    print(
        diagnostics[
            "grad_theta"
        ]
    )


    # ========================================================
    # VERY IMPORTANT:
    # These are NOT the same quantity
    # ========================================================

    print(
        "\nActual PRECONDITIONED flow RHS:"
    )

    print(
        diagnostics[
            "flow_rhs"
        ]
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
        "stationarity residual:"
    )

    print(
        diagnostics[
            "stationarity_residual"
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
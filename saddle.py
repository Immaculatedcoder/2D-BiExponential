import numpy as np
from scipy.integrate import solve_ivp

from varpro import (
    build_design_matrix,
    build_design_matrix_derivatives,
)

# ============================================================
# Equality-constraint matrix
#
# h(c) = 0.5 * c.T @ H @ c
#
# with
#
# c = [k1, k1_hat, k2, k2_hat]
#
# so that
#
# h(c) = k1_hat*k2 - k1*k2_hat
# ============================================================

H = np.array([
    [ 0.0,  0.0,  0.0, -1.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  1.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0,  0.0],
])

# ============================================================
# Inequality-constraint matrix
#
# A.T @ c + q <= 0
#
# Constraints:
#
#   k1 >= eps_k
#   k1_hat >= 0
#   k1_hat <= 2*k1
#
#   k2 >= eps_k
#   k2_hat >= 0
#   k2_hat <= 2*k2
# ============================================================

A = np.array([
    [-1.0,  0.0, -2.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  1.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0, -1.0,  0.0, -2.0],
    [ 0.0,  0.0,  0.0,  0.0, -1.0,  1.0],
])

def build_q(eps_k=1e-8):
    """
    Build the constant vector q for

        A.T @ c + q <= 0.

    The small positive eps_k enforces

        k1 >= eps_k equivalently k1>0
        k2 >= eps_k.
    """

    q = np.array([
        eps_k,
        0.0,
        0.0,
        eps_k,
        0.0,
        0.0
    ])

    return q

def build_B(TI, TE, theta, lam):
    """
        B(theta, lambda) = Phi(theta).T @ Phi(theta) + lambda * H

        Returns
        -------
        B : ndarray, shape (4,4)
    """
    Phi = build_design_matrix(
        TI,
        TE,
        theta
    )

    B = Phi.T @ Phi + lam * H

    return B

def solve_reduced_c(TI, TE, theta, mu, lam, d):
    """
        recover c* = argmin L(c, theta, mu, lam)

        B(theta, lambda) c* = Phi(theta).T @ d - A @ mu

        Parameters
        ----------

        Returns
        c : ndarray, shape (4,)
            Reduced optimal linear coefficients
    """

    Phi = build_design_matrix(TI, TE, theta)

    B = Phi.T @ Phi + lam * H
    rhs = Phi.T @ d - A @ mu

    c = np.linalg.solve(B, rhs)

    return c

def reduced_quantities(TI, TE, theta, mu, lam, d, eps_k=1e-8):
    """
        We esstially just put the reduced saddle quantites as one.

        c* : reduced linear coefficients
        r  : data residual
        g  : inequality-constraint residual
        h  : equality-constraint residual
    """

    c = solve_reduced_c(TI, TE, theta, mu, lam, d)

    Phi = build_design_matrix(TI, TE, theta)

    r = Phi @ c - d

    q = build_q(eps_k)
    g = A.T @ c + q

    h = 0.5 * c @ H @ c

    return c, r, g, h 

def reduced_gradients(TI, TE, theta, mu, lam, d, eps_k=1e-8):
    """
    Returns:
        1. grad_theta
        2. grad_mu
        3. grad_lam
    """
    c, r, g, h = reduced_quantities(TI, TE, theta, mu, lam, d, eps_k=eps_k)

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

    # Gradient with respect to theta
    grad_theta = np.array([
        r @ (dPhi_dT11 @ c),
        r @ (dPhi_dT12 @ c),
        r @ (dPhi_dT21 @ c),
        r @ (dPhi_dT22 @ c)
    ])

    # Gradient with respect to mu
    grad_mu = g

    # Gradient with respect to lambda
    grad_lam = h

    return grad_theta, grad_mu, grad_lam

def project_mu_direction(mu, direction, tol=1e-12):
    """
    Project the dual ascent direction so that mu>=0 is preserved

    direction: Unprojected ascent direction
    """
    projected = direction.copy()

    for i in range(len(mu)):
        if mu[i] <= tol and direction[i] < 0.0:
            projected[i] = 0.0

    return projected

def project_theta_direction(theta, direction, theta_lower, theta_upper, tol=1e-12):
    """
    """

    projected = direction.copy()

    for j in range(len(theta)):
        # At lower bound: don't try to push to the left
        if (theta[j] <= theta_lower[j] + tol and direction[j] < 0.0):
            projected[j] = 0.0

        # At upper bound: don't try to push to the right
        elif (theta[j] >= theta_upper[j] - tol and direction[j] > 0.0):
            projected[j] = 0.0

    return projected

def saddle_rhs(t, y, TI, TE, d, theta_lower, theta_upper, gamma_theta=1.0,gamma_mu=1.0,gamma_lam=1.0, eps_k=1e-8):
    """
    Arrow-Hurwicz-Uzawa saddle dynamics

    State vector
    y =[theta, mu, lambda]

    Dynamics
    --------
        theta_dot  = projected descent direction
        mu_dot     = projected ascent direction
        lambda_dot = equality-multiplier ascent direction
    """

    # Unpack the state vector
    theta = y[0:4]
    mu = y[4:10]
    lam=y[10]

    # Compute the reduced gradients
    grad_theta, grad_mu, grad_lam = reduced_gradients(TI, TE, theta, mu, lam, d, eps_k=1e-8)

    # Descent direction for theta
    theta_direction = -gamma_theta * grad_theta
    theta_dot = project_theta_direction(theta, theta_direction, theta_lower, theta_upper)

    # ascent direction for mu
    mu_direction = gamma_mu * grad_mu

    mu_dot = project_mu_direction(mu, mu_direction)

    # Equality mulitplier
    lambda_dot = gamma_lam * grad_lam

    # Now we can pack them back together
    y_dot = np.concatenate([
        theta_dot,
        mu_dot,
        np.array([lambda_dot])
    ])

    return y_dot


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
    rtol=1e-8,
    atol=1e-10,
    max_step=np.inf,
    b_eig_tol=1e-8,
    steady_tol = 1e-5,
):
    """
    Our goal here is to integrate in time
    """

    # 1. Convert initial values to array
    theta0 = np.asarray(theta0, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)

    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)

    # 2-3. Check theta feasibility
    if np.any(theta0 < theta_lower) or np.any(theta0 > theta_upper):
        raise ValueError(
            "Initial theta must lie inside the theta bounds."
        )

    # 4. Check dual feasibility
    if np.any(mu0 < 0.0):
        raise ValueError(
            "Initial mu must satisfy mu >= 0."
        )

    # 5. Check initial B matrix
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
            "Initial B matrix is not sufficiently "
            "positive definite."
        )

    # 5. Build initial state vector
    y0 = np.concatenate([
        theta0,
        mu0,
        np.array([lam0], dtype=float)
    ])

    # 6. RHS
    def rhs(t, y):

        return saddle_rhs(
            t,
            y,
            TI,
            TE,
            d,
            theta_lower,
            theta_upper,
            gamma_theta,
            gamma_mu,
            gamma_lam,
            eps_k
        )

    # Event to stop: stop if B approaches loss positive definiteness
    def b_positive_definite_event(t, y):

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

        return min_eig - b_eig_tol


    b_positive_definite_event.terminal = True
    b_positive_definite_event.direction = -1  # Goes from positive to negative

    def steady_state_event(t, y):

        y_dot = rhs(t, y)

        rhs_norm = np.linalg.norm(y_dot)

        return rhs_norm - steady_tol

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
            b_positive_definite_event, steady_state_event
        ]
    )

    return solution

# -----------------------------------------------
# Preconditioning
# ------------------------------------------
def build_theta_preconditioner(
    theta_lower,
    theta_upper
):
    """
    Build a normalized range-based diagonal
    preconditioner for theta.

    theta_scale = theta_upper - theta_lower

    Raw preconditioner:
        P = diag(theta_scale^2)

    The matrix is normalized by its largest
    diagonal entry.
    """

    theta_scale = theta_upper - theta_lower

    weights = theta_scale**2

    weights = weights / np.max(weights)

    P_theta = np.diag(weights)

    return P_theta

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
    Arrow-Hurwicz-Uzawa saddle dynamics.

    State:
        y = [theta, mu, lambda]
    """

    # --------------------------------------------------
    # 1. Unpack state
    # --------------------------------------------------

    theta = y[0:4]
    mu = y[4:10]
    lam = y[10]


    # --------------------------------------------------
    # 2. Reduced gradients
    # --------------------------------------------------

    grad_theta, grad_mu, grad_lam = reduced_gradients(
        TI,
        TE,
        theta,
        mu,
        lam,
        d,
        eps_k
    )


    # --------------------------------------------------
    # 3. Default mu preconditioner
    # --------------------------------------------------

    if P_mu is None:
        P_mu = np.eye(6)


    # --------------------------------------------------
    # 4. Preconditioned theta descent
    #
    # theta_dot = -gamma_theta P_theta grad_theta
    # --------------------------------------------------

    theta_direction = (
        -gamma_theta
        * (P_theta @ grad_theta)
    )

    theta_dot = project_theta_direction(
        theta,
        theta_direction,
        theta_lower,
        theta_upper
    )


    # --------------------------------------------------
    # 5. Preconditioned mu ascent
    #
    # mu_dot = gamma_mu P_mu grad_mu
    # --------------------------------------------------

    mu_direction = (
        gamma_mu
        * (P_mu @ grad_mu)
    )

    mu_dot = project_mu_direction(
        mu,
        mu_direction
    )


    # --------------------------------------------------
    # 6. Preconditioned lambda ascent
    # --------------------------------------------------

    lambda_dot = (
        gamma_lam
        * p_lam
        * grad_lam
    )


    # --------------------------------------------------
    # 7. Assemble state derivative
    # --------------------------------------------------

    y_dot = np.concatenate([
        theta_dot,
        mu_dot,
        np.array([lambda_dot])
    ])

    return y_dot

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
    Integrate the preconditioned reduced
    saddle dynamics.
    """

    theta0 = np.asarray(theta0, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)

    theta_lower = np.asarray(theta_lower, dtype=float)
    theta_upper = np.asarray(theta_upper, dtype=float)

    P_theta = np.asarray(P_theta, dtype=float)

    if P_mu is None:
        P_mu = np.eye(6)

    P_mu = np.asarray(P_mu, dtype=float)


    # --------------------------------------------------
    # Initial-condition checks
    # --------------------------------------------------

    if theta0.shape != (4,):
        raise ValueError(
            "theta0 must have shape (4,)."
        )

    if mu0.shape != (6,):
        raise ValueError(
            "mu0 must have shape (6,)."
        )

    if P_theta.shape != (4, 4):
        raise ValueError(
            "P_theta must have shape (4, 4)."
        )

    if P_mu.shape != (6, 6):
        raise ValueError(
            "P_mu must have shape (6, 6)."
        )

    if (
        np.any(theta0 < theta_lower)
        or np.any(theta0 > theta_upper)
    ):
        raise ValueError(
            "Initial theta must lie inside the theta bounds."
        )

    if np.any(mu0 < 0.0):
        raise ValueError(
            "Initial mu must satisfy mu >= 0."
        )


    # --------------------------------------------------
    # Check initial B
    # --------------------------------------------------

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
            "Initial B matrix is not sufficiently "
            "positive definite."
        )


    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------

    y0 = np.concatenate([
        theta0,
        mu0,
        np.array([lam0])
    ])


    # --------------------------------------------------
    # RHS
    # --------------------------------------------------

    def rhs(t, y):

        return saddle_rhs_preconditioned(
            t,
            y,
            TI,
            TE,
            d,
            theta_lower,
            theta_upper,
            P_theta,
            P_mu,
            p_lam,
            gamma_theta,
            gamma_mu,
            gamma_lam,
            eps_k
        )


    # --------------------------------------------------
    # B positive-definite event
    # --------------------------------------------------

    def b_positive_definite_event(t, y):

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

        return min_eig - b_eig_tol


    b_positive_definite_event.terminal = True
    b_positive_definite_event.direction = -1


    # --------------------------------------------------
    # Steady-state event
    # --------------------------------------------------

    def steady_state_event(t, y):

        y_dot = rhs(t, y)

        return (
            np.linalg.norm(y_dot)
            - steady_tol
        )


    steady_state_event.terminal = True
    steady_state_event.direction = -1


    # --------------------------------------------------
    # Integrate
    # --------------------------------------------------

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


if __name__ == "__main__":

    # c_true = np.array([
    #     120.0,
    #     234.0,
    #     80.0,
    #     156.0
    # ])

    # q = build_q()

    # g = A.T @ c_true + q

    # h = 0.5 * c_true @ H @ c_true

    # print("\nInequality residuals:")
    # print(g)

    # print("\nEquality residual:")
    # print(h)

    from config import TI, TE, TRUE_PARAMS, THETA_LOWER, THETA_UPPER, THETA0

    theta_true = np.array([
        TRUE_PARAMS["T11"],
        TRUE_PARAMS["T12"],
        TRUE_PARAMS["T21"],
        TRUE_PARAMS["T22"]
    ])

    lam = 0.0

    # B = build_B(TI, TE, theta_true, lam)

    # print("B shape:")
    # print(B.shape)

    # print("\nB:")
    # print(B)

    # print("\nSymmetry error:")
    # print(np.linalg.norm(B - B.T))

    # print("\nEigenvalues:")
    # print(np.linalg.eigvalsh(B))

    # print("\nCondition number:")
    # print(np.linalg.cond(B))
    # -----------------------------------------

    from model import biexponential_2d
    mu = np.zeros(6)

    D_clean = biexponential_2d(TI, TE, TRUE_PARAMS)
    d_clean = D_clean.ravel(order="F")

    c, r, g, h = reduced_quantities(
        TI,
        TE,
        theta_true,
        np.zeros(6),
        0.0,
        d_clean
    )

    print("c:")
    print(c)

    print("\nResidual norm:")
    print(np.linalg.norm(r))

    print("\nInequality residuals g:")
    print(g)

    print("\nEquality residual h:")
    print(h)

    # ----------------------------------
    # grad_theta, grad_mu, grad_lam = reduced_gradients(
    #     TI,
    #     TE,
    #     theta_true,
    #     np.zeros(6),
    #     0.0,
    #     d_clean
    # )

    # print("grad_theta:")
    # print(grad_theta)

    # print("\ngrad_mu:")
    # print(grad_mu)

    # print("\ngrad_lambda:")
    # print(grad_lam)

    # --------------------------------------
    # mu = np.array([
    #     0.0,
    #     0.0,
    #     1.0,
    #     0.5,
    #     0.0,
    #     2.0
    # ])

    # direction = np.array([
    #     -3.0,
    #     2.0,
    #     -1.0,
    #     4.0,
    #     -5.0,
    #     -2.0
    # ])

    # projected = project_mu_direction(
    #     mu,
    #     direction
    # )

    # print(projected)

    # ------------
    mu0 = np.zeros(6)

    lam0 = 0.0
    # y_true = np.concatenate([
    #     theta_true,
    #     mu0,
    #     np.array([lam0])
    # ])

    # y_dot = saddle_rhs(
    #     0.0,
    #     y_true,
    #     TI,
    #     TE,
    #     d_clean,
    #     THETA_LOWER,
    #     THETA_UPPER
    # )

    # print("y_dot:")
    # print(y_dot)

    # print("\nRHS norm:")
    # print(np.linalg.norm(y_dot))


    # --------------------- Unpreconditioned
    # solution = solve_saddle(
    #     THETA0,
    #     np.zeros(6),
    #     0.0,
    #     TI,
    #     TE,
    #     d_clean,
    #     THETA_LOWER,
    #     THETA_UPPER,

    #     t_span=(0.0, 5000.0),

    #     gamma_theta=1.0,
    #     gamma_mu=1.0,
    #     gamma_lam=1.0,

    #     method="BDF",

    #     rtol=1e-6,
    #     atol=1e-8,

    #     max_step=5.0,

    #     steady_tol=1e-5
    # )

    # -----Preconditioned

    P_theta = build_theta_preconditioner(
        THETA_LOWER,
        THETA_UPPER
    )

    print("P_theta:")
    print(P_theta)

    solution = solve_saddle_preconditioned(
        THETA0,
        np.zeros(6),
        0.0,
        TI,
        TE,
        d_clean,
        THETA_LOWER,
        THETA_UPPER,

        t_span=(0.0, 5000.0),

        P_theta=P_theta,

        P_mu=np.eye(6),
        p_lam=1.0,

        gamma_theta=1.0,
        gamma_mu=1.0,
        gamma_lam=1.0,

        method="BDF",

        rtol=1e-6,
        atol=1e-8,

        max_step=20.0,

        steady_tol=1e-5
    )

    print("Success:", solution.success)
    print("Message:", solution.message)

    print("Initial time:", solution.t[0])
    print("Final time:", solution.t[-1])

    print("Number of stored time points:")
    print(len(solution.t))

    y_final = solution.y[:, -1]

    theta_final = y_final[0:4]
    mu_final = y_final[4:10]
    lam_final = y_final[10]


    # Recover reduced quantities
    c_final, r_final, g_final, h_final = reduced_quantities(
        TI,
        TE,
        theta_final,
        mu_final,
        lam_final,
        d_clean
    )


    # Evaluate final RHS
    # ydot_final = saddle_rhs(
    #     solution.t[-1],
    #     y_final,
    #     TI,
    #     TE,
    #     d_clean,
    #     THETA_LOWER,
    #     THETA_UPPER
    # )

    ydot_final = saddle_rhs_preconditioned(
        solution.t[-1],
        y_final,
        TI,
        TE,
        d_clean,
        THETA_LOWER,
        THETA_UPPER,
        P_theta,
        P_mu=np.eye(6),
        p_lam=1.0,
        gamma_theta=1.0,
        gamma_mu=1.0,
        gamma_lam=1.0
    )

    print("\nPreconditioned final RHS:")
    print(ydot_final)

    print("\nPreconditioned final RHS norm:")
    print(np.linalg.norm(ydot_final))

    # Build B
    B_final = build_B(
        TI,
        TE,
        theta_final,
        lam_final
    )


    print("Final theta:")
    print(theta_final)

    print("\nFinal c:")
    print(c_final)

    print("\nFinal beta:")
    print("beta1 =", c_final[1] / c_final[0])
    print("beta2 =", c_final[3] / c_final[2])

    print("\nResidual norm:")
    print(np.linalg.norm(r_final))

    print("\nInequality residuals:")
    print(g_final)

    print("\nEquality residual h:")
    print(h_final)

    print("\nFinal RHS:")
    print(ydot_final)

    print("\nFinal RHS norm:")
    print(np.linalg.norm(ydot_final))

    print("\nMinimum eigenvalue of B:")
    print(np.min(np.linalg.eigvalsh(B_final)))

    print("\nCondition number of B:")
    print(np.linalg.cond(B_final))

    grad_theta, grad_mu, grad_lam = reduced_gradients(
        TI,
        TE,
        theta_final,
        mu_final,
        lam_final,
        d_clean
    )

    print("Final grad_theta:")
    print(grad_theta)

    print("\nIndividual theta gradients:")
    print("dL/dT11 =", grad_theta[0])
    print("dL/dT12 =", grad_theta[1])
    print("dL/dT21 =", grad_theta[2])
    print("dL/dT22 =", grad_theta[3])

    c, r, g, h = reduced_quantities(
        TI,
        TE,
        theta_final,
        mu_final,
        lam_final,
        d_clean
    )

    dPhi_dT11, dPhi_dT12, dPhi_dT21, dPhi_dT22 = \
        build_design_matrix_derivatives(
            TI,
            TE,
            theta_final
        )

    print("T11 term:", r @ (dPhi_dT11 @ c))
    print("T12 term:", r @ (dPhi_dT12 @ c))
    print("T21 term:", r @ (dPhi_dT21 @ c))
    print("T22 term:", r @ (dPhi_dT22 @ c))

    print("B-event times:")
    print(solution.t_events[0])

    print("\nSteady-state event times:")
    print(solution.t_events[1])

    print("Success:", solution.success)
    print("Message:", solution.message)

    print("Final time:", solution.t[-1])

    print("Stored time points:", len(solution.t))
    print("Function evaluations:", solution.nfev)

    print("B-event:")
    print(solution.t_events[0])

    print("Steady-state event:")
    print(solution.t_events[1])

    print("Jacobian evaluations:", solution.njev)
    print("LU decompositions:", solution.nlu)
"""
    We create a Dataset and then run the complete Architecture

    This program takes any data set and then Gives the parameters.
"""

import numpy as np

from optimizer import solve_theta
from varpro import (
    build_design_matrix,
    solve_linear_coefficients
)

def estimate_parameters(D, TI, TE, theta0, theta_lower, theta_upper):
    """
        Returns
        k1, k2, beta1, beta2, T11, T12, T21, T22
    """
    # Step 1: Vectorize/Flatten the Data(mxn)-> mnx1
    d = D.ravel(order="F")

    # Step 2: Sove the nonlinear VarPro Problem
    result = solve_theta(theta0, TI, TE, d, theta_lower, theta_upper)
    theta_est = result.x

    # Step 3: Recover the optimal linear coefficients
    Phi_est = build_design_matrix(TI, TE, theta_est)
    c_est = solve_linear_coefficients(Phi_est, d)

    # Step 4: Recover Physical parameters
    k1 = c_est[0]
    k1_hat = c_est[1]

    k2 = c_est[2]
    k2_hat = c_est[3]

    beta1 = k1_hat / k1
    beta2 = k2_hat / k2

    # Step 5: Bundle the result we have
    estimate = {
        "k1": k1,
        "k2": k2,
        "beta1": beta1,
        "beta2": beta2,
        "T11": theta_est[0],
        "T12": theta_est[1],
        "T21": theta_est[2],
        "T22": theta_est[3],
    }

    return estimate, result
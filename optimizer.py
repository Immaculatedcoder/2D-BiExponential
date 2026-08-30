from scipy.optimize import minimize, Bounds

from varpro import varpro_objective, varpro_gradient


"""
    constrained Solution
"""
def solve_theta(theta0, TI, TE, d, theta_lower, theta_upper):
    """
    Solve the constrained optimization 

        min_theta F(theta)
    
    subject to
        theta_lower <= theta <= theta_upper
    
    where: theta = [T11, T12, T21, T22]
    """

    bounds = Bounds(
        lb=theta_lower,
        ub=theta_upper
    )

    result = minimize(
        varpro_objective, 
        theta0,
        args=(TI, TE, d),
        jac=varpro_gradient,
        method="L-BFGS-B",
        bounds=bounds
    )

    return result

# Test Begins-----
# from config import TI, TE, TRUE_PARAMS, THETA0
# from model import biexponential_2d
# from varpro import build_design_matrix, solve_linear_coefficients

# D_clean = biexponential_2d(
#     TI,
#     TE,
#     TRUE_PARAMS
# )

# d_clean = D_clean.ravel(order="F")

# result = solve_theta(
#     THETA0,
#     TI,
#     TE,
#     d_clean,
#     theta_lower,
#     theta_upper
# )

# Phi_est = build_design_matrix(TI, TE, result.x)
# c_est = solve_linear_coefficients(
#     Phi_est,
#     d_clean
# )

# k1_est = c_est[0]
# beta1_est = c_est[1] / c_est[0]

# k2_est = c_est[2]
# beta2_est = c_est[3] / c_est[2]


# print(result.x)
# print("k1    =", k1_est)
# print("beta1 =", beta1_est)
# print("k2    =", k2_est)
# print("beta2 =", beta2_est)
# print(result.nit)
# print(result.nfev)

# Test Ends----
import numpy as np

from config import (
    TI,
    TE,
    TRUE_PARAMS,
    THETA0,
    THETA_LOWER,
    THETA_UPPER,
    SIGMA,
    RANDOM_SEED
)

from model import biexponential_2d
from data import add_gaussian_noise
from estimate import estimate_parameters

# ============================================================
# True parameter vector for comparison
# ============================================================

PARAMETER_NAMES = [
    "k1",
    "k2",
    "beta1",
    "beta2",
    "T11",
    "T12",
    "T21",
    "T22",
]

def print_results(title, estimates, result):
    """
    Print estimated parameters and errors.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\nOptimizer success:")
    print(result.success)

    print("\nOptimizer message:")
    print(result.message)

    print("\nParameter estimates:")
    print("-" * 60)

    for name in PARAMETER_NAMES:
        true_value = TRUE_PARAMS[name]
        estimated_value = estimates[name]

        absolute_error = abs(
            estimated_value - true_value
        )

        relative_error = (
            absolute_error / abs(true_value)
        ) 

        print(
            f"{name:6s} "
            f"true = {true_value:12.6f}   "
            f"estimated = {estimated_value:12.6f}   "
            f"rel. error = {relative_error:10.6f}%"
        )

    print("\nOuter optimizer diagnostics:")
    print("Iterations:", result.nit)
    print("Function evaluations:", result.nfev)

    if hasattr(result, "njev"):
        print("Gradient evaluations:", result.njev)

    print("Final objective:", result.fun)

# ============================================================
# 1. CLEAN DATA TEST
# ============================================================

D_clean = biexponential_2d(
    TI,
    TE,
    TRUE_PARAMS
)

clean_estimates, clean_result = estimate_parameters(
    D_clean,
    TI,
    TE,
    THETA0,
    THETA_LOWER,
    THETA_UPPER
)

print_results(
    "CLEAN DATA TEST",
    clean_estimates,
    clean_result
)

# ============================================================
# 2. NOISY DATA TEST
# ============================================================

D_noisy = add_gaussian_noise(
    D_clean,
    SIGMA,
    RANDOM_SEED
)

noisy_estimates, noisy_result = estimate_parameters(
    D_noisy,
    TI,
    TE,
    THETA0,
    THETA_LOWER,
    THETA_UPPER
)

print_results(
    "NOISY DATA TEST",
    noisy_estimates,
    noisy_result
)


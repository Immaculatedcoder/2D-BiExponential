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
# CLEAN DATA
# ============================================================

D_clean = biexponential_2d(
    TI,
    TE,
    TRUE_PARAMS
)


estimate_clean, result_clean = estimate_parameters(
    D_clean,
    TI,
    TE,
    THETA0,
    THETA_LOWER,
    THETA_UPPER
)


print(
    "\n================================"
)

print(
    "SCIPY — CLEAN DATA"
)

print(
    "================================"
)

for key, value in estimate_clean.items():
    print(
        f"{key:6s} = {value}"
    )


print(
    "\nbeta1 - beta2 =",
    estimate_clean["beta1"]
    - estimate_clean["beta2"]
)

print(
    "\nOuter success:",
    result_clean.success
)

print(
    "Outer message:",
    result_clean.message
)


# ============================================================
# NOISY DATA
# ============================================================

D_noisy = add_gaussian_noise(
    D_clean,
    sigma=SIGMA,
    seed=RANDOM_SEED
)


estimate_noisy, result_noisy = estimate_parameters(
    D_noisy,
    TI,
    TE,
    THETA0,
    THETA_LOWER,
    THETA_UPPER
)


print(
    "\n================================"
)

print(
    "SCIPY — NOISY DATA"
)

print(
    "================================"
)

for key, value in estimate_noisy.items():
    print(
        f"{key:6s} = {value}"
    )


print(
    "\nbeta1 - beta2 =",
    estimate_noisy["beta1"]
    - estimate_noisy["beta2"]
)

print(
    "\nOuter success:",
    result_noisy.success
)

print(
    "Outer message:",
    result_noisy.message
)
import numpy as np

from config import (
    TRUE_PARAMS,
    THETA0
)

from plotting_mc import (
    plot_all_solver_comparisons,
    print_solver_summary
)


# ============================================================
# TEST DATA ONLY
#
# These are artificial distributions.
# They are NOT results from either solver.
#
# The purpose is only to test:
#
#   1. two-solver plotting
#   2. common histogram bins
#   3. true-value lines
#   4. solver means
#   5. summary statistics
# ============================================================

rng = np.random.default_rng(42)

N_test = 500


# ============================================================
# Fake SciPy VarPro results
# ============================================================

scipy_results = {

    "k1": rng.normal(
        loc=TRUE_PARAMS["k1"],
        scale=0.40,
        size=N_test
    ),

    "beta1": rng.normal(
        loc=TRUE_PARAMS["beta1"],
        scale=0.006,
        size=N_test
    ),

    "k2": rng.normal(
        loc=TRUE_PARAMS["k2"],
        scale=0.35,
        size=N_test
    ),

    "beta2": rng.normal(
        loc=TRUE_PARAMS["beta2"],
        scale=0.006,
        size=N_test
    ),

    "T11": rng.normal(
        loc=TRUE_PARAMS["T11"],
        scale=4.0,
        size=N_test
    ),

    "T12": rng.normal(
        loc=TRUE_PARAMS["T12"],
        scale=9.0,
        size=N_test
    ),

    "T21": rng.normal(
        loc=TRUE_PARAMS["T21"],
        scale=0.35,
        size=N_test
    ),

    "T22": rng.normal(
        loc=TRUE_PARAMS["T22"],
        scale=0.80,
        size=N_test
    ),
}


# ============================================================
# Fake Preconditioned Saddle results
#
# Deliberately make these slightly different so that
# we can clearly see two distributions.
# ============================================================

saddle_results = {

    "k1": rng.normal(
        loc=TRUE_PARAMS["k1"] + 0.05,
        scale=0.50,
        size=N_test
    ),

    "beta1": rng.normal(
        loc=TRUE_PARAMS["beta1"] - 0.0005,
        scale=0.007,
        size=N_test
    ),

    "k2": rng.normal(
        loc=TRUE_PARAMS["k2"] - 0.03,
        scale=0.45,
        size=N_test
    ),

    "beta2": rng.normal(
        loc=TRUE_PARAMS["beta2"] - 0.0005,
        scale=0.007,
        size=N_test
    ),

    "T11": rng.normal(
        loc=TRUE_PARAMS["T11"] + 0.5,
        scale=5.0,
        size=N_test
    ),

    "T12": rng.normal(
        loc=TRUE_PARAMS["T12"] + 1.0,
        scale=11.0,
        size=N_test
    ),

    "T21": rng.normal(
        loc=TRUE_PARAMS["T21"] + 0.02,
        scale=0.45,
        size=N_test
    ),

    "T22": rng.normal(
        loc=TRUE_PARAMS["T22"] + 0.10,
        scale=1.0,
        size=N_test
    ),
}


# ============================================================
# Combine results using FINAL Monte Carlo structure
# ============================================================

results_by_solver = {

    "SciPy VarPro":
        scipy_results,

    "Preconditioned Saddle":
        saddle_results
}


# ============================================================
# Print statistical summary
# ============================================================

print_solver_summary(
    results_by_solver,
    TRUE_PARAMS,
    THETA0
)


# ============================================================
# Generate comparison plots
# ============================================================

plot_all_solver_comparisons(

    results_by_solver=
        results_by_solver,

    true_params=
        TRUE_PARAMS,

    theta0=
        THETA0,

    bins=30,

    output_dir=
        "mc_two_solver_test_plots",

    show=True,

    density=True
)


print(
    "\n============================================"
)

print(
    "TWO-SOLVER PLOTTING TEST COMPLETE"
)

print(
    "============================================"
)

print(
    "\nFigures saved in:"
)

print(
    "mc_two_solver_test_plots/"
)
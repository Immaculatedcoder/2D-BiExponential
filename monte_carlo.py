import csv
import os
import time
from pathlib import Path

import numpy as np


# ============================================================
# Project imports
# ============================================================

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

# SciPy VarPro solver
from estimate import estimate_parameters

# Preconditioned saddle solver
from saddle2 import (
    build_theta_preconditioner,
    solve_saddle_preconditioned,
    saddle_diagnostics
)

# Plotting/statistics module
from plotting_mc import (
    plot_all_solver_comparisons,
    print_solver_summary
)


# ============================================================
# Experiment settings
# ============================================================

N_REALIZATIONS = 200

BASE_SEED = RANDOM_SEED

STEADY_TOL = 1e-5

T_SPAN = (
    0.0,
    10000.0
)

GAMMA_THETA = 1.0
GAMMA_MU = 1.0
GAMMA_LAM = 1.0

P_LAM = 1.0

BDF_RTOL = 1e-6
BDF_ATOL = 1e-8
BDF_MAX_STEP = 20.0


# ============================================================
# Output settings
# ============================================================

OUTPUT_DIR = Path(
    "mc_results"
)

PLOT_DIR = OUTPUT_DIR / "plots"

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / f"checkpoint_N{N_REALIZATIONS}.npz"
)

CSV_FILE = (
    OUTPUT_DIR
    / f"monte_carlo_N{N_REALIZATIONS}.csv"
)

ERROR_LOG = (
    OUTPUT_DIR
    / f"errors_N{N_REALIZATIONS}.txt"
)


# If True:
# reload an existing checkpoint and continue where we stopped.
RESUME = True

# For N=5 the histograms are NOT statistically meaningful.
# They are only checking that the entire pipeline works.
GENERATE_PLOTS = True


# ============================================================
# Solver names
#
# Keep these EXACTLY the same as the plotting code.
# ============================================================

SCIPY_NAME = "SciPy"

SADDLE_NAME = "AHU-VarPro-Saddle"

SOLVER_NAMES = [
    SCIPY_NAME,
    SADDLE_NAME
]


# ============================================================
# Physical parameters to store
# ============================================================

PARAMETER_NAMES = [
    "k1",
    "beta1",
    "k2",
    "beta2",
    "T11",
    "T12",
    "T21",
    "T22",
]


# ============================================================
# Diagnostic names
# ============================================================

SCIPY_DIAGNOSTIC_NAMES = [
    "runtime",
    "nfev",
    "nit",
    "objective"
]

SADDLE_DIAGNOSTIC_NAMES = [
    "runtime",
    "nfev",
    "njev",
    "nlu",
    "final_time",
    "stationarity_norm",
    "flow_rhs_norm",
    "residual_norm",
    "h",
    "lambda",
    "B_min_eigenvalue",
    "B_condition_number"
]


# ============================================================
# Initialize storage
# ============================================================

def initialize_storage(N):
    """
    Allocate storage for all Monte Carlo realizations.

    Each parameter array has length N.

    Failed realizations remain NaN.
    """

    results = {

        SCIPY_NAME: {
            name: np.full(
                N,
                np.nan,
                dtype=float
            )
            for name in PARAMETER_NAMES
        },

        SADDLE_NAME: {
            name: np.full(
                N,
                np.nan,
                dtype=float
            )
            for name in PARAMETER_NAMES
        }
    }


    diagnostics = {

        SCIPY_NAME: {

            "success":
                np.zeros(
                    N,
                    dtype=bool
                ),

            **{
                name: np.full(
                    N,
                    np.nan,
                    dtype=float
                )
                for name
                in SCIPY_DIAGNOSTIC_NAMES
            }
        },


        SADDLE_NAME: {

            "success":
                np.zeros(
                    N,
                    dtype=bool
                ),

            **{
                name: np.full(
                    N,
                    np.nan,
                    dtype=float
                )
                for name
                in SADDLE_DIAGNOSTIC_NAMES
            }
        }
    }


    completed = np.zeros(
        N,
        dtype=bool
    )


    seeds = np.arange(
        BASE_SEED,
        BASE_SEED + N,
        dtype=int
    )


    return (
        results,
        diagnostics,
        completed,
        seeds
    )


# ============================================================
# Store one physical estimate
# ============================================================

def store_estimate(
    results,
    solver_name,
    index,
    estimate
):
    """
    Store one solver's physical parameter estimates.
    """

    for name in PARAMETER_NAMES:

        results[
            solver_name
        ][name][index] = float(
            estimate[name]
        )


# ============================================================
# Convert saddle diagnostics into physical parameters
# ============================================================

def saddle_physical_estimate(
    diagnostics
):
    """
    Convert saddle2 diagnostics into the same physical
    parameter dictionary returned by estimate_parameters().
    """

    theta = diagnostics[
        "theta"
    ]

    c = diagnostics[
        "c"
    ]

    estimate = {

        "k1":
            float(c[0]),

        "beta1":
            float(
                c[1] / c[0]
            ),

        "k2":
            float(c[2]),

        "beta2":
            float(
                c[3] / c[2]
            ),

        "T11":
            float(theta[0]),

        "T12":
            float(theta[1]),

        "T21":
            float(theta[2]),

        "T22":
            float(theta[3])
    }

    return estimate


# ============================================================
# Run SciPy VarPro on ONE realization
# ============================================================

def run_scipy_realization(
    D_noisy
):
    """
    Run the equality-constrained SciPy VarPro solver.
    """

    start = time.perf_counter()

    estimate, result = estimate_parameters(
        D_noisy,
        TI,
        TE,
        THETA0,
        THETA_LOWER,
        THETA_UPPER
    )

    runtime = (
        time.perf_counter()
        - start
    )


    if not result.success:

        raise RuntimeError(
            "SciPy outer optimization failed: "
            f"{result.message}"
        )


    info = {

        "runtime":
            runtime,

        "nfev":
            float(
                getattr(
                    result,
                    "nfev",
                    np.nan
                )
            ),

        "nit":
            float(
                getattr(
                    result,
                    "nit",
                    np.nan
                )
            ),

        "objective":
            float(
                getattr(
                    result,
                    "fun",
                    np.nan
                )
            )
    }


    return estimate, info


# ============================================================
# Run preconditioned saddle solver on ONE realization
# ============================================================

def run_saddle_realization(
    D_noisy,
    P_theta,
    P_mu
):
    """
    Run the validated preconditioned reduced saddle solver.
    """

    d_noisy = D_noisy.ravel(
        order="F"
    )


    start = time.perf_counter()


    solution = solve_saddle_preconditioned(

        THETA0,

        np.zeros(6),

        0.0,

        TI,
        TE,

        d_noisy,

        THETA_LOWER,
        THETA_UPPER,

        t_span=T_SPAN,

        P_theta=P_theta,

        P_mu=P_mu,

        p_lam=P_LAM,

        gamma_theta=GAMMA_THETA,

        gamma_mu=GAMMA_MU,

        gamma_lam=GAMMA_LAM,

        method="BDF",

        rtol=BDF_RTOL,

        atol=BDF_ATOL,

        max_step=BDF_MAX_STEP,

        steady_tol=STEADY_TOL
    )


    runtime = (
        time.perf_counter()
        - start
    )


    # --------------------------------------------------------
    # Full final diagnostics
    # --------------------------------------------------------

    diag = saddle_diagnostics(

        solution,

        TI,
        TE,

        d_noisy,

        THETA_LOWER,
        THETA_UPPER,

        preconditioned=True,

        P_theta=P_theta,

        P_mu=P_mu,

        p_lam=P_LAM,

        gamma_theta=GAMMA_THETA,

        gamma_mu=GAMMA_MU,

        gamma_lam=GAMMA_LAM
    )


    # --------------------------------------------------------
    # Check solve_ivp status
    # --------------------------------------------------------

    if not solution.success:

        raise RuntimeError(
            "Saddle ODE integration failed: "
            f"{solution.message}"
        )


    # --------------------------------------------------------
    # Check B-event
    #
    # If this fired, B approached loss of
    # positive definiteness.
    # --------------------------------------------------------

    B_event_occurred = (
        len(solution.t_events[0])
        > 0
    )

    if B_event_occurred:

        raise RuntimeError(
            "Saddle solver terminated because "
            "B approached loss of positive definiteness."
        )


    # --------------------------------------------------------
    # Check true stationarity
    #
    # We accept either:
    #
    #   1. steady-state event fired
    #
    # or
    #
    #   2. final stationarity norm is numerically
    #      within tolerance.
    # --------------------------------------------------------

    steady_event_occurred = (
        len(solution.t_events[1])
        > 0
    )


    stationarity_norm = float(
        diag[
            "stationarity_norm"
        ]
    )


    stationary = (

        steady_event_occurred

        or

        (
            stationarity_norm
            <= 1.01 * STEADY_TOL
        )
    )


    if not stationary:

        raise RuntimeError(
            "Saddle solver reached the end of t_span "
            "without satisfying stationarity. "
            f"Final stationarity norm = "
            f"{stationarity_norm:.6e}"
        )


    # --------------------------------------------------------
    # Physical estimates
    # --------------------------------------------------------

    estimate = (
        saddle_physical_estimate(
            diag
        )
    )


    # --------------------------------------------------------
    # Solver diagnostics
    # --------------------------------------------------------

    info = {

        "runtime":
            runtime,

        "nfev":
            float(
                solution.nfev
            ),

        "njev":
            float(
                solution.njev
            ),

        "nlu":
            float(
                solution.nlu
            ),

        "final_time":
            float(
                solution.t[-1]
            ),

        "stationarity_norm":
            stationarity_norm,

        "flow_rhs_norm":
            float(
                diag[
                    "flow_rhs_norm"
                ]
            ),

        "residual_norm":
            float(
                diag[
                    "residual_norm"
                ]
            ),

        "h":
            float(
                diag["h"]
            ),

        "lambda":
            float(
                diag["lambda"]
            ),

        "B_min_eigenvalue":
            float(
                diag[
                    "B_min_eigenvalue"
                ]
            ),

        "B_condition_number":
            float(
                diag[
                    "B_condition_number"
                ]
            )
    }


    return estimate, info


# ============================================================
# Store diagnostics
# ============================================================

def store_diagnostics(
    diagnostics,
    solver_name,
    index,
    info
):
    """
    Store one realization's solver diagnostics.
    """

    diagnostics[
        solver_name
    ]["success"][index] = True


    for key, value in info.items():

        if key in diagnostics[
            solver_name
        ]:

            diagnostics[
                solver_name
            ][key][index] = float(
                value
            )


# ============================================================
# Error logging
# ============================================================

def log_error(
    realization,
    seed,
    solver_name,
    error
):
    """
    Append failures to a text log without stopping the
    entire Monte Carlo experiment.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        ERROR_LOG,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Realization {realization + 1}, "
            f"seed={seed}, "
            f"solver={solver_name}\n"
        )

        f.write(
            f"{type(error).__name__}: "
            f"{error}\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )


# ============================================================
# Checkpoint handling
# ============================================================

SOLVER_PREFIX = {
    SCIPY_NAME: "scipy",
    SADDLE_NAME: "saddle"
}


def save_checkpoint(
    results,
    diagnostics,
    completed,
    seeds
):
    """
    Save the entire current Monte Carlo state.

    This is called after EVERY realization.

    Therefore if the program is interrupted, previous
    realizations are not lost.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    payload = {

        "N":
            np.array(
                N_REALIZATIONS,
                dtype=int
            ),

        "base_seed":
            np.array(
                BASE_SEED,
                dtype=int
            ),

        "completed":
            completed,

        "seeds":
            seeds
    }


    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    for solver_name in SOLVER_NAMES:

        prefix = SOLVER_PREFIX[
            solver_name
        ]

        for parameter_name in PARAMETER_NAMES:

            key = (
                f"{prefix}"
                f"__param__"
                f"{parameter_name}"
            )

            payload[key] = results[
                solver_name
            ][parameter_name]


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    for solver_name in SOLVER_NAMES:

        prefix = SOLVER_PREFIX[
            solver_name
        ]

        for diagnostic_name, values in (
            diagnostics[
                solver_name
            ].items()
        ):

            key = (
                f"{prefix}"
                f"__diag__"
                f"{diagnostic_name}"
            )

            payload[key] = values


    # --------------------------------------------------------
    # Atomic save
    #
    # Write temporary file first, then replace checkpoint.
    # --------------------------------------------------------

    temporary_file = (
        str(CHECKPOINT_FILE)
        + ".tmp"
    )


    with open(
        temporary_file,
        "wb"
    ) as f:

        np.savez_compressed(
            f,
            **payload
        )


    os.replace(
        temporary_file,
        CHECKPOINT_FILE
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint():
    """
    Load previous Monte Carlo state.

    Returns None if there is no checkpoint.
    """

    if not CHECKPOINT_FILE.exists():

        return None


    (
        results,
        diagnostics,
        completed,
        seeds
    ) = initialize_storage(
        N_REALIZATIONS
    )


    with np.load(
        CHECKPOINT_FILE,
        allow_pickle=False
    ) as data:


        saved_N = int(
            data["N"]
        )


        saved_seed = int(
            data["base_seed"]
        )


        if saved_N != N_REALIZATIONS:

            raise ValueError(
                "Checkpoint was created with "
                f"N={saved_N}, but current "
                f"N_REALIZATIONS={N_REALIZATIONS}."
            )


        if saved_seed != BASE_SEED:

            raise ValueError(
                "Checkpoint uses a different "
                "BASE_SEED."
            )


        completed[:] = data[
            "completed"
        ]

        seeds[:] = data[
            "seeds"
        ]


        # ----------------------------------------------------
        # Restore parameters
        # ----------------------------------------------------

        for solver_name in SOLVER_NAMES:

            prefix = SOLVER_PREFIX[
                solver_name
            ]

            for parameter_name in PARAMETER_NAMES:

                key = (
                    f"{prefix}"
                    f"__param__"
                    f"{parameter_name}"
                )

                results[
                    solver_name
                ][parameter_name][:] = (
                    data[key]
                )


        # ----------------------------------------------------
        # Restore diagnostics
        # ----------------------------------------------------

        for solver_name in SOLVER_NAMES:

            prefix = SOLVER_PREFIX[
                solver_name
            ]

            for diagnostic_name in (
                diagnostics[
                    solver_name
                ].keys()
            ):

                key = (
                    f"{prefix}"
                    f"__diag__"
                    f"{diagnostic_name}"
                )

                diagnostics[
                    solver_name
                ][diagnostic_name][:] = (
                    data[key]
                )


    return (
        results,
        diagnostics,
        completed,
        seeds
    )


# ============================================================
# Build results containing ONLY paired successes
# ============================================================

def build_paired_results(
    results,
    diagnostics
):
    """
    Keep only realizations for which BOTH solvers succeeded.

    This is the preferred dataset for comparing parameter
    distributions because both solvers then see exactly the
    same collection of noise realizations.
    """

    paired_mask = (

        diagnostics[
            SCIPY_NAME
        ]["success"]

        &

        diagnostics[
            SADDLE_NAME
        ]["success"]
    )


    paired_results = {

        SCIPY_NAME: {},

        SADDLE_NAME: {}
    }


    for solver_name in SOLVER_NAMES:

        for parameter_name in PARAMETER_NAMES:

            paired_results[
                solver_name
            ][parameter_name] = (

                results[
                    solver_name
                ][parameter_name][
                    paired_mask
                ]
            )


    return (
        paired_results,
        paired_mask
    )


# ============================================================
# Save per-realization CSV
# ============================================================

def save_csv(
    results,
    diagnostics,
    seeds
):
    """
    Save one row per Monte Carlo realization.

    This is useful later for statistical analysis,
    debugging and thesis tables.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    fieldnames = [
        "realization",
        "seed",
        "scipy_success",
        "saddle_success"
    ]


    for parameter_name in PARAMETER_NAMES:

        fieldnames.append(
            f"scipy_{parameter_name}"
        )

        fieldnames.append(
            f"saddle_{parameter_name}"
        )


    fieldnames.extend([
        "scipy_runtime",
        "scipy_nfev",
        "scipy_nit",
        "scipy_objective",

        "saddle_runtime",
        "saddle_nfev",
        "saddle_njev",
        "saddle_nlu",
        "saddle_final_time",
        "saddle_stationarity_norm",
        "saddle_residual_norm",
        "saddle_h",
        "saddle_lambda",
        "saddle_B_min_eigenvalue",
        "saddle_B_condition_number"
    ])


    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )


        writer.writeheader()


        for i in range(
            N_REALIZATIONS
        ):


            row = {

                "realization":
                    i + 1,

                "seed":
                    int(
                        seeds[i]
                    ),

                "scipy_success":
                    bool(
                        diagnostics[
                            SCIPY_NAME
                        ]["success"][i]
                    ),

                "saddle_success":
                    bool(
                        diagnostics[
                            SADDLE_NAME
                        ]["success"][i]
                    )
            }


            for parameter_name in PARAMETER_NAMES:

                row[
                    f"scipy_{parameter_name}"
                ] = results[
                    SCIPY_NAME
                ][parameter_name][i]


                row[
                    f"saddle_{parameter_name}"
                ] = results[
                    SADDLE_NAME
                ][parameter_name][i]


            # SciPy diagnostics
            row["scipy_runtime"] = (
                diagnostics[
                    SCIPY_NAME
                ]["runtime"][i]
            )

            row["scipy_nfev"] = (
                diagnostics[
                    SCIPY_NAME
                ]["nfev"][i]
            )

            row["scipy_nit"] = (
                diagnostics[
                    SCIPY_NAME
                ]["nit"][i]
            )

            row["scipy_objective"] = (
                diagnostics[
                    SCIPY_NAME
                ]["objective"][i]
            )


            # Saddle diagnostics
            for key in [
                "runtime",
                "nfev",
                "njev",
                "nlu",
                "final_time",
                "stationarity_norm",
                "residual_norm",
                "h",
                "lambda",
                "B_min_eigenvalue",
                "B_condition_number"
            ]:

                row[
                    f"saddle_{key}"
                ] = diagnostics[
                    SADDLE_NAME
                ][key][i]


            writer.writerow(
                row
            )


# ============================================================
# Monte Carlo engine
# ============================================================

def run_monte_carlo():
    """
    Run paired Monte Carlo realizations.

    For realization i:

        1. generate ONE noisy dataset D_i
        2. send D_i to SciPy
        3. send SAME D_i to saddle solver
        4. store both estimates
        5. checkpoint immediately
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Clean truth
    # --------------------------------------------------------

    D_clean = biexponential_2d(
        TI,
        TE,
        TRUE_PARAMS
    )


    # --------------------------------------------------------
    # Preconditioners are identical for every realization
    # --------------------------------------------------------

    P_theta = build_theta_preconditioner(
        THETA_LOWER,
        THETA_UPPER
    )

    P_mu = np.eye(6)


    # --------------------------------------------------------
    # Resume or initialize
    # --------------------------------------------------------

    checkpoint = None


    if RESUME:

        checkpoint = (
            load_checkpoint()
        )


    if checkpoint is None:

        (
            results,
            diagnostics,
            completed,
            seeds
        ) = initialize_storage(
            N_REALIZATIONS
        )

    else:

        (
            results,
            diagnostics,
            completed,
            seeds
        ) = checkpoint

        print(
            "\nExisting checkpoint loaded."
        )

        print(
            "Completed realizations:",
            int(
                np.sum(completed)
            ),
            "/",
            N_REALIZATIONS
        )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        "PAIRED MONTE CARLO EXPERIMENT"
    )

    print(
        "============================================================"
    )

    print(
        "\nN realizations:",
        N_REALIZATIONS
    )

    print(
        "Noise sigma:",
        SIGMA
    )

    print(
        "Base seed:",
        BASE_SEED
    )

    print(
        "\nTrue theta:"
    )

    print(
        np.array([
            TRUE_PARAMS["T11"],
            TRUE_PARAMS["T12"],
            TRUE_PARAMS["T21"],
            TRUE_PARAMS["T22"]
        ])
    )

    print(
        "\nInitial theta:"
    )

    print(
        THETA0
    )

    print(
        "\nP_theta:"
    )

    print(
        P_theta
    )


    experiment_start = (
        time.perf_counter()
    )


    # ========================================================
    # Main realization loop
    # ========================================================

    for i in range(
        N_REALIZATIONS
    ):


        if completed[i]:

            print(
                f"\nRealization {i + 1}/{N_REALIZATIONS} "
                "already completed. Skipping."
            )

            continue


        seed = int(
            seeds[i]
        )


        print(
            "\n============================================================"
        )

        print(
            f"REALIZATION {i + 1}/{N_REALIZATIONS}"
        )

        print(
            "============================================================"
        )

        print(
            "Seed:",
            seed
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Generate noisy data ONCE.
        #
        # Both solvers receive this exact D_noisy.
        # ----------------------------------------------------

        D_noisy = add_gaussian_noise(
            D_clean,
            sigma=SIGMA,
            seed=seed
        )


        # ====================================================
        # SciPy VarPro
        # ====================================================

        print(
            "\nRunning SciPy VarPro..."
        )


        try:

            scipy_estimate, scipy_info = (
                run_scipy_realization(
                    D_noisy
                )
            )


            store_estimate(
                results,
                SCIPY_NAME,
                i,
                scipy_estimate
            )


            store_diagnostics(
                diagnostics,
                SCIPY_NAME,
                i,
                scipy_info
            )


            print(
                "SciPy success."
            )

            print(
                "  theta =",
                np.array([
                    scipy_estimate["T11"],
                    scipy_estimate["T12"],
                    scipy_estimate["T21"],
                    scipy_estimate["T22"]
                ])
            )

            print(
                "  beta  =",
                scipy_estimate["beta1"]
            )

            print(
                "  runtime =",
                f"{scipy_info['runtime']:.3f} s"
            )


        except Exception as error:

            print(
                "SciPy FAILED:"
            )

            print(
                error
            )


            log_error(
                i,
                seed,
                SCIPY_NAME,
                error
            )


        # ====================================================
        # Preconditioned saddle
        # ====================================================

        print(
            "\nRunning Preconditioned Saddle..."
        )


        try:

            saddle_estimate, saddle_info = (
                run_saddle_realization(
                    D_noisy,
                    P_theta,
                    P_mu
                )
            )


            store_estimate(
                results,
                SADDLE_NAME,
                i,
                saddle_estimate
            )


            store_diagnostics(
                diagnostics,
                SADDLE_NAME,
                i,
                saddle_info
            )


            print(
                "Saddle success."
            )

            print(
                "  theta =",
                np.array([
                    saddle_estimate["T11"],
                    saddle_estimate["T12"],
                    saddle_estimate["T21"],
                    saddle_estimate["T22"]
                ])
            )

            print(
                "  beta  =",
                saddle_estimate["beta1"]
            )

            print(
                "  stationarity =",
                saddle_info[
                    "stationarity_norm"
                ]
            )

            print(
                "  runtime =",
                f"{saddle_info['runtime']:.3f} s"
            )


        except Exception as error:

            print(
                "Saddle FAILED:"
            )

            print(
                error
            )


            log_error(
                i,
                seed,
                SADDLE_NAME,
                error
            )


        # ====================================================
        # Paired comparison for this realization
        # ====================================================

        if (

            diagnostics[
                SCIPY_NAME
            ]["success"][i]

            and

            diagnostics[
                SADDLE_NAME
            ]["success"][i]
        ):


            scipy_theta = np.array([
                results[SCIPY_NAME]["T11"][i],
                results[SCIPY_NAME]["T12"][i],
                results[SCIPY_NAME]["T21"][i],
                results[SCIPY_NAME]["T22"][i]
            ])


            saddle_theta = np.array([
                results[SADDLE_NAME]["T11"][i],
                results[SADDLE_NAME]["T12"][i],
                results[SADDLE_NAME]["T21"][i],
                results[SADDLE_NAME]["T22"][i]
            ])


            theta_difference = (
                scipy_theta
                - saddle_theta
            )


            print(
                "\nPaired solver comparison:"
            )

            print(
                "  SciPy theta - Saddle theta ="
            )

            print(
                theta_difference
            )

            print(
                "  max |theta difference| =",
                np.max(
                    np.abs(
                        theta_difference
                    )
                )
            )


        # ----------------------------------------------------
        # Mark realization attempted
        # ----------------------------------------------------

        completed[i] = True


        # ----------------------------------------------------
        # SAVE AFTER EVERY REALIZATION
        # ----------------------------------------------------

        save_checkpoint(
            results,
            diagnostics,
            completed,
            seeds
        )


        save_csv(
            results,
            diagnostics,
            seeds
        )


        print(
            "\nCheckpoint saved."
        )


    # ========================================================
    # End of Monte Carlo loop
    # ========================================================

    total_runtime = (
        time.perf_counter()
        - experiment_start
    )


    scipy_successes = int(
        np.sum(
            diagnostics[
                SCIPY_NAME
            ]["success"]
        )
    )


    saddle_successes = int(
        np.sum(
            diagnostics[
                SADDLE_NAME
            ]["success"]
        )
    )


    paired_results, paired_mask = (
        build_paired_results(
            results,
            diagnostics
        )
    )


    paired_successes = int(
        np.sum(
            paired_mask
        )
    )


    print(
        "\n============================================================"
    )

    print(
        "MONTE CARLO COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        "\nTotal wall-clock time:",
        f"{total_runtime:.3f} s"
    )

    print(
        "\nSciPy successes:",
        scipy_successes,
        "/",
        N_REALIZATIONS
    )

    print(
        "Saddle successes:",
        saddle_successes,
        "/",
        N_REALIZATIONS
    )

    print(
        "Paired successes:",
        paired_successes,
        "/",
        N_REALIZATIONS
    )


    # ========================================================
    # Paired statistical summary
    # ========================================================

    if paired_successes > 0:

        print(
            "\n============================================================"
        )

        print(
            "PAIRED MONTE CARLO STATISTICS"
        )

        print(
            "============================================================"
        )


        print_solver_summary(
            paired_results,
            TRUE_PARAMS,
            THETA0
        )


        # ====================================================
        # Plots
        # ====================================================

        if GENERATE_PLOTS:

            plot_all_solver_comparisons(

                results_by_solver=
                    paired_results,

                true_params=
                    TRUE_PARAMS,

                theta0=
                    THETA0,

                bins=30,

                output_dir=
                    PLOT_DIR,

                show=True,

                density=True
            )


    else:

        print(
            "\nNo paired successful realizations."
        )

        print(
            "Plots/statistics were not generated."
        )


    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_FILE
    )

    print(
        "\nCSV results:"
    )

    print(
        CSV_FILE
    )

    print(
        "\nPlots:"
    )

    print(
        PLOT_DIR
    )


    return (
        results,
        diagnostics,
        paired_results
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_monte_carlo()
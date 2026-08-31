"""
monte_carlo_equal_beta.py

Paired Monte Carlo comparison:

    1. Equal-Beta SciPy VarPro
    2. AHU-VarPro-Saddle

The same noisy data realization is sent to both solvers.

This file is intentionally separate from the previous monte_carlo.py so
that the earlier baseline experiment is preserved.

Required project files
----------------------
    config.py
    model.py
    varpro.py
    saddle2.py
    plotting_mc.py

    varpro_equal_beta.py
    optimizer_equal_beta.py
    estimate_equal_beta.py

Important formulation note
--------------------------
Both methods in THIS experiment impose

    beta1 = beta2.

For Equal-Beta SciPy VarPro this is imposed by reparameterization:

    alpha = [T11, T12, T21, T22, beta]

with linear coefficients

    k = [k1, k2].

For AHU-VarPro-Saddle the equality is imposed through the existing
saddle formulation.

All data vectorization uses order="F".
"""

from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path

import numpy as np

from config import (
    TI,
    TE,
    TRUE_PARAMS,
    SIGMA,
    RANDOM_SEED,
    THETA0,
    THETA_LOWER,
    THETA_UPPER,
)

from model import biexponential_2d

from estimate_equal_beta import (
    estimate_parameters_equal_beta,
)

from saddle2 import (
    build_theta_preconditioner,
    solve_saddle_preconditioned,
    reduced_quantities,
    reduced_gradients,
    project_theta_direction,
    project_mu_direction,
    saddle_rhs_preconditioned,
    build_B,
)

from plotting_mc import (
    plot_all_solver_comparisons,
    print_solver_summary,
)


# ============================================================
# Experiment settings
# ============================================================

# Start small first. After validation, change to 20, 200, 2000, etc.
N_REALIZATIONS = 5

BASE_SEED = int(RANDOM_SEED)

# ------------------------------------------------------------
# Equal-beta SciPy settings
# ------------------------------------------------------------

BETA0 = 1.0
BETA_LOWER = 0.0
BETA_UPPER = 2.0
EPS_K = 1e-8

SCIPY_OPTIONS = {
    "maxiter": 3000,
    "ftol": 1e-12,
    "gtol": 1e-10,
    "maxls": 50,
}

# ------------------------------------------------------------
# AHU-VarPro-Saddle settings
#
# These match the preconditioned configuration that has already
# been used successfully in the current project.
# ------------------------------------------------------------

MU0 = np.zeros(6, dtype=float)
LAM0 = 0.0

T_SPAN = (0.0, 10000.0)

P_MU = np.eye(6, dtype=float)
P_LAM = 1.0

GAMMA_THETA = 1.0
GAMMA_MU = 1.0
GAMMA_LAM = 1.0

SADDLE_METHOD = "BDF"
SADDLE_RTOL = 1e-6
SADDLE_ATOL = 1e-8
SADDLE_MAX_STEP = 20.0
SADDLE_STEADY_TOL = 1e-5
B_EIG_TOL = 1e-8

# ------------------------------------------------------------
# Output settings
#
# DO NOT use the old mc_results folder.  This experiment has a
# different SciPy formulation and must have its own checkpoint.
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "mc_results_equal_beta_vs_ahu"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / f"checkpoint_equal_beta_N{N_REALIZATIONS}.npz"
)

CSV_FILE = (
    OUTPUT_DIR
    / f"monte_carlo_equal_beta_N{N_REALIZATIONS}.csv"
)

ERROR_LOG_FILE = (
    OUTPUT_DIR
    / f"errors_equal_beta_N{N_REALIZATIONS}.txt"
)

RESUME = True
GENERATE_PLOTS = True


# ============================================================
# Solver display names
# ============================================================

SCIPY_NAME = "Equal-Beta SciPy VarPro"
SADDLE_NAME = "AHU-VarPro-Saddle"


# ============================================================
# Physical parameter ordering
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
# True parameters for plotting / summaries
# ============================================================

TRUE_PHYSICAL_PARAMS = {
    name: float(TRUE_PARAMS[name])
    for name in PARAMETER_NAMES
}


# ============================================================
# Preconditioner
# ============================================================

P_THETA = build_theta_preconditioner(
    THETA_LOWER,
    THETA_UPPER,
)


# ============================================================
# Utility: create empty result arrays
# ============================================================

def make_empty_parameter_results():
    """
    Allocate NaN-filled arrays for all physical parameters.
    """

    return {
        name: np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        )
        for name in PARAMETER_NAMES
    }


def make_empty_diagnostics():
    """
    Diagnostics stored realization-by-realization.
    """

    return {
        # Common
        "success": np.zeros(
            N_REALIZATIONS,
            dtype=bool,
        ),
        "runtime": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),

        # SciPy
        "nit": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "nfev": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "objective": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),

        # Saddle
        "njev": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "nlu": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "final_time": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "stationarity_norm": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "flow_rhs_norm": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "residual_norm": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "h": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "lambda": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "B_min_eig": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
        "B_cond": np.full(
            N_REALIZATIONS,
            np.nan,
            dtype=float,
        ),
    }


# ============================================================
# Utility: store one physical parameter dictionary
# ============================================================

def store_parameter_estimate(
    results,
    index,
    estimate,
):
    for name in PARAMETER_NAMES:
        results[name][index] = float(
            estimate[name]
        )


# ============================================================
# Utility: generate one paired noisy dataset
# ============================================================

def generate_noisy_dataset(
    clean_signal,
    seed,
):
    """
    Additive Gaussian noise:

        D = D_clean + epsilon,
        epsilon_ij ~ N(0, SIGMA^2).
    """

    rng = np.random.default_rng(
        int(seed)
    )

    noise = rng.normal(
        loc=0.0,
        scale=float(SIGMA),
        size=clean_signal.shape,
    )

    return clean_signal + noise


# ============================================================
# Utility: convert saddle c, theta to physical parameters
# ============================================================

def saddle_physical_parameters(
    c,
    theta,
):
    c = np.asarray(
        c,
        dtype=float,
    )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    if c.shape != (4,):
        raise ValueError(
            "Saddle coefficient vector c must have shape (4,)."
        )

    if theta.shape != (4,):
        raise ValueError(
            "Saddle theta must have shape (4,)."
        )

    k1 = float(c[0])
    k1_hat = float(c[1])

    k2 = float(c[2])
    k2_hat = float(c[3])

    if k1 <= 0.0 or k2 <= 0.0:
        raise RuntimeError(
            "Saddle returned nonpositive k1 or k2."
        )

    beta1 = k1_hat / k1
    beta2 = k2_hat / k2

    return {
        "k1": k1,
        "beta1": float(beta1),
        "k2": k2,
        "beta2": float(beta2),
        "T11": float(theta[0]),
        "T12": float(theta[1]),
        "T21": float(theta[2]),
        "T22": float(theta[3]),
    }


# ============================================================
# Preconditioner-independent projected stationarity residual
# ============================================================

def projected_stationarity_residual(
    theta,
    mu,
    lam,
    d,
):
    """
    Compute a convergence diagnostic that does NOT contain
    P_theta, P_mu, p_lam, or the gamma step multipliers.

    For the reduced saddle problem:

        theta : projected descent direction -grad_theta
        mu    : projected ascent direction  grad_mu
        lambda: grad_lambda

    This is useful for comparing convergence independently of
    the chosen preconditioner.
    """

    grad_theta, grad_mu, grad_lam = (
        reduced_gradients(
            TI,
            TE,
            theta,
            mu,
            lam,
            d,
            EPS_K,
        )
    )

    theta_direction = (
        -np.asarray(
            grad_theta,
            dtype=float,
        )
    )

    theta_projected = (
        project_theta_direction(
            np.asarray(theta, dtype=float),
            theta_direction,
            np.asarray(
                THETA_LOWER,
                dtype=float,
            ),
            np.asarray(
                THETA_UPPER,
                dtype=float,
            ),
        )
    )

    mu_direction = np.asarray(
        grad_mu,
        dtype=float,
    )

    mu_projected = (
        project_mu_direction(
            np.asarray(mu, dtype=float),
            mu_direction,
        )
    )

    return np.concatenate([
        theta_projected,
        mu_projected,
        np.array(
            [float(grad_lam)],
            dtype=float,
        ),
    ])


# ============================================================
# Run Equal-Beta SciPy once
# ============================================================

def run_equal_beta_scipy(
    D_noisy,
):
    start = time.perf_counter()

    estimate, result = (
        estimate_parameters_equal_beta(
            D_noisy,
            TI,
            TE,
            THETA0,
            THETA_LOWER,
            THETA_UPPER,
            beta0=BETA0,
            beta_lower=BETA_LOWER,
            beta_upper=BETA_UPPER,
            eps_k=EPS_K,
            options=SCIPY_OPTIONS,
        )
    )

    runtime = (
        time.perf_counter()
        - start
    )

    theta = np.array(
        [
            estimate["T11"],
            estimate["T12"],
            estimate["T21"],
            estimate["T22"],
        ],
        dtype=float,
    )

    diagnostics = {
        "runtime": float(runtime),
        "nit": float(
            getattr(
                result,
                "nit",
                np.nan,
            )
        ),
        "nfev": float(
            getattr(
                result,
                "nfev",
                np.nan,
            )
        ),
        "objective": float(
            result.fun
        ),
        "theta": theta,
        "beta": float(
            estimate["beta1"]
        ),
    }

    return estimate, result, diagnostics


# ============================================================
# Run AHU-VarPro-Saddle once
# ============================================================

def run_ahu_saddle(
    D_noisy,
):
    d = np.asarray(
        D_noisy,
        dtype=float,
    ).ravel(
        order="F"
    )

    start = time.perf_counter()

    solution = solve_saddle_preconditioned(
        THETA0,
        MU0,
        LAM0,
        TI,
        TE,
        d,
        THETA_LOWER,
        THETA_UPPER,
        t_span=T_SPAN,
        P_theta=P_THETA,
        P_mu=P_MU,
        p_lam=P_LAM,
        gamma_theta=GAMMA_THETA,
        gamma_mu=GAMMA_MU,
        gamma_lam=GAMMA_LAM,
        eps_k=EPS_K,
        method=SADDLE_METHOD,
        rtol=SADDLE_RTOL,
        atol=SADDLE_ATOL,
        max_step=SADDLE_MAX_STEP,
        b_eig_tol=B_EIG_TOL,
        steady_tol=SADDLE_STEADY_TOL,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    if not solution.success:
        raise RuntimeError(
            "AHU-VarPro-Saddle integration failed: "
            f"{solution.message}"
        )

    y_final = np.asarray(
        solution.y[:, -1],
        dtype=float,
    )

    theta_final = y_final[0:4]
    mu_final = y_final[4:10]
    lam_final = float(
        y_final[10]
    )

    c_final, r_final, g_final, h_final = (
        reduced_quantities(
            TI,
            TE,
            theta_final,
            mu_final,
            lam_final,
            d,
            EPS_K,
        )
    )

    estimate = saddle_physical_parameters(
        c_final,
        theta_final,
    )

    raw_stationarity = (
        projected_stationarity_residual(
            theta_final,
            mu_final,
            lam_final,
            d,
        )
    )

    flow_rhs = (
        saddle_rhs_preconditioned(
            float(
                solution.t[-1]
            ),
            y_final,
            TI,
            TE,
            d,
            THETA_LOWER,
            THETA_UPPER,
            P_THETA,
            P_MU,
            P_LAM,
            GAMMA_THETA,
            GAMMA_MU,
            GAMMA_LAM,
            EPS_K,
        )
    )

    B_final = build_B(
        TI,
        TE,
        theta_final,
        lam_final,
    )

    eigvals = np.linalg.eigvalsh(
        B_final
    )

    diagnostics = {
        "runtime": float(runtime),
        "nfev": float(
            getattr(
                solution,
                "nfev",
                np.nan,
            )
        ),
        "njev": float(
            getattr(
                solution,
                "njev",
                np.nan,
            )
        ),
        "nlu": float(
            getattr(
                solution,
                "nlu",
                np.nan,
            )
        ),
        "final_time": float(
            solution.t[-1]
        ),
        "stationarity_norm": float(
            np.linalg.norm(
                raw_stationarity
            )
        ),
        "flow_rhs_norm": float(
            np.linalg.norm(
                flow_rhs
            )
        ),
        "residual_norm": float(
            np.linalg.norm(
                r_final
            )
        ),
        "h": float(
            h_final
        ),
        "lambda": float(
            lam_final
        ),
        "B_min_eig": float(
            np.min(
                eigvals
            )
        ),
        "B_cond": float(
            np.linalg.cond(
                B_final
            )
        ),
        "theta": theta_final.copy(),
        "mu": mu_final.copy(),
        "c": np.asarray(
            c_final,
            dtype=float,
        ).copy(),
        "g": np.asarray(
            g_final,
            dtype=float,
        ).copy(),
    }

    return estimate, solution, diagnostics


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    completed,
    scipy_results,
    saddle_results,
    scipy_diag,
    saddle_diag,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "completed": completed.astype(
            np.int8
        ),
    }

    for name in PARAMETER_NAMES:
        payload[
            f"scipy_param_{name}"
        ] = scipy_results[name]

        payload[
            f"saddle_param_{name}"
        ] = saddle_results[name]

    for key, values in scipy_diag.items():
        if isinstance(
            values,
            np.ndarray,
        ):
            payload[
                f"scipy_diag_{key}"
            ] = values

    for key, values in saddle_diag.items():
        if isinstance(
            values,
            np.ndarray,
        ):
            payload[
                f"saddle_diag_{key}"
            ] = values

    np.savez(
        CHECKPOINT_FILE,
        **payload,
    )


def load_checkpoint(
    completed,
    scipy_results,
    saddle_results,
    scipy_diag,
    saddle_diag,
):
    if not (
        RESUME
        and CHECKPOINT_FILE.exists()
    ):
        return

    data = np.load(
        CHECKPOINT_FILE,
        allow_pickle=False,
    )

    if "completed" in data:
        loaded = np.asarray(
            data["completed"],
            dtype=bool,
        )

        if loaded.shape != completed.shape:
            raise RuntimeError(
                "Checkpoint size does not match "
                "N_REALIZATIONS. Use a new output "
                "folder/checkpoint for this run."
            )

        completed[:] = loaded

    for name in PARAMETER_NAMES:
        key = f"scipy_param_{name}"

        if key in data:
            scipy_results[name][:] = data[
                key
            ]

        key = f"saddle_param_{name}"

        if key in data:
            saddle_results[name][:] = data[
                key
            ]

    for diag_name, diag in [
        ("scipy", scipy_diag),
        ("saddle", saddle_diag),
    ]:

        for key in diag.keys():

            file_key = (
                f"{diag_name}_diag_{key}"
            )

            if file_key in data:
                diag[key][:] = data[
                    file_key
                ]

    print(
        "\nLoaded checkpoint:"
    )

    print(
        CHECKPOINT_FILE
    )

    print(
        "Completed realizations:",
        int(
            np.sum(
                completed
            )
        ),
        "/",
        N_REALIZATIONS,
    )


# ============================================================
# Error log
# ============================================================

def log_error(
    realization_index,
    seed,
    solver_name,
    exc,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ERROR_LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        handle.write(
            f"Realization: {realization_index + 1}\n"
        )

        handle.write(
            f"Seed: {seed}\n"
        )

        handle.write(
            f"Solver: {solver_name}\n"
        )

        handle.write(
            f"Error: {repr(exc)}\n\n"
        )

        handle.write(
            traceback.format_exc()
        )

        handle.write(
            "\n"
        )


# ============================================================
# CSV writer
# ============================================================

def write_csv(
    seeds,
    completed,
    scipy_results,
    saddle_results,
    scipy_diag,
    saddle_diag,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "realization",
        "seed",
        "completed",
        "scipy_success",
        "saddle_success",
        "paired_success",
    ]

    for name in PARAMETER_NAMES:
        fieldnames.extend([
            f"scipy_{name}",
            f"saddle_{name}",
            f"paired_diff_{name}",
        ])

    fieldnames.extend([
        "scipy_runtime",
        "scipy_nit",
        "scipy_nfev",
        "scipy_objective",
        "saddle_runtime",
        "saddle_nfev",
        "saddle_njev",
        "saddle_nlu",
        "saddle_final_time",
        "saddle_stationarity_norm",
        "saddle_flow_rhs_norm",
        "saddle_residual_norm",
        "saddle_h",
        "saddle_lambda",
        "saddle_B_min_eig",
        "saddle_B_cond",
    ])

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(
            N_REALIZATIONS
        ):

            scipy_success = bool(
                scipy_diag["success"][i]
            )

            saddle_success = bool(
                saddle_diag["success"][i]
            )

            paired_success = (
                scipy_success
                and saddle_success
            )

            row = {
                "realization": i + 1,
                "seed": int(
                    seeds[i]
                ),
                "completed": bool(
                    completed[i]
                ),
                "scipy_success": scipy_success,
                "saddle_success": saddle_success,
                "paired_success": paired_success,
            }

            for name in PARAMETER_NAMES:

                scipy_value = float(
                    scipy_results[
                        name
                    ][i]
                )

                saddle_value = float(
                    saddle_results[
                        name
                    ][i]
                )

                row[
                    f"scipy_{name}"
                ] = scipy_value

                row[
                    f"saddle_{name}"
                ] = saddle_value

                if (
                    np.isfinite(
                        scipy_value
                    )
                    and np.isfinite(
                        saddle_value
                    )
                ):
                    row[
                        f"paired_diff_{name}"
                    ] = (
                        scipy_value
                        - saddle_value
                    )
                else:
                    row[
                        f"paired_diff_{name}"
                    ] = np.nan

            row.update({
                "scipy_runtime":
                    scipy_diag["runtime"][i],
                "scipy_nit":
                    scipy_diag["nit"][i],
                "scipy_nfev":
                    scipy_diag["nfev"][i],
                "scipy_objective":
                    scipy_diag["objective"][i],

                "saddle_runtime":
                    saddle_diag["runtime"][i],
                "saddle_nfev":
                    saddle_diag["nfev"][i],
                "saddle_njev":
                    saddle_diag["njev"][i],
                "saddle_nlu":
                    saddle_diag["nlu"][i],
                "saddle_final_time":
                    saddle_diag["final_time"][i],
                "saddle_stationarity_norm":
                    saddle_diag[
                        "stationarity_norm"
                    ][i],
                "saddle_flow_rhs_norm":
                    saddle_diag[
                        "flow_rhs_norm"
                    ][i],
                "saddle_residual_norm":
                    saddle_diag[
                        "residual_norm"
                    ][i],
                "saddle_h":
                    saddle_diag["h"][i],
                "saddle_lambda":
                    saddle_diag["lambda"][i],
                "saddle_B_min_eig":
                    saddle_diag[
                        "B_min_eig"
                    ][i],
                "saddle_B_cond":
                    saddle_diag[
                        "B_cond"
                    ][i],
            })

            writer.writerow(
                row
            )


# ============================================================
# Keep only realizations where BOTH solvers succeeded
# ============================================================

def build_paired_results(
    scipy_results,
    saddle_results,
    scipy_diag,
    saddle_diag,
):
    paired_mask = (
        scipy_diag["success"]
        & saddle_diag["success"]
    )

    paired_results = {
        SCIPY_NAME: {},
        SADDLE_NAME: {},
    }

    for name in PARAMETER_NAMES:

        paired_results[
            SCIPY_NAME
        ][name] = (
            scipy_results[
                name
            ][paired_mask]
        )

        paired_results[
            SADDLE_NAME
        ][name] = (
            saddle_results[
                name
            ][paired_mask]
        )

    return paired_results, paired_mask


# ============================================================
# Main Monte Carlo loop
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 68
    )

    print(
        "EQUAL-BETA SCIPY VARPRO vs AHU-VARPRO-SADDLE"
    )

    print(
        "=" * 68
    )

    print(
        "\nN_REALIZATIONS =",
        N_REALIZATIONS,
    )

    print(
        "SIGMA =",
        SIGMA,
    )

    print(
        "THETA0 =",
        np.asarray(
            THETA0,
            dtype=float,
        ),
    )

    print(
        "Equal-beta SciPy beta0 =",
        BETA0,
    )

    print(
        "Equal-beta bounds =",
        (
            BETA_LOWER,
            BETA_UPPER,
        ),
    )

    print(
        "\nP_theta:"
    )

    print(
        P_THETA
    )

    # --------------------------------------------------------
    # Clean synthetic signal
    # --------------------------------------------------------

    D_clean = biexponential_2d(
        TI,
        TE,
        TRUE_PARAMS,
    )

    expected_shape = (
        len(TI),
        len(TE),
    )

    if D_clean.shape != expected_shape:
        raise RuntimeError(
            f"Clean model returned shape "
            f"{D_clean.shape}; expected "
            f"{expected_shape}."
        )

    # --------------------------------------------------------
    # Allocate storage
    # --------------------------------------------------------

    scipy_results = (
        make_empty_parameter_results()
    )

    saddle_results = (
        make_empty_parameter_results()
    )

    scipy_diag = (
        make_empty_diagnostics()
    )

    saddle_diag = (
        make_empty_diagnostics()
    )

    completed = np.zeros(
        N_REALIZATIONS,
        dtype=bool,
    )

    seeds = (
        BASE_SEED
        + np.arange(
            N_REALIZATIONS,
            dtype=int,
        )
    )

    # --------------------------------------------------------
    # Resume if a checkpoint exists
    # --------------------------------------------------------

    load_checkpoint(
        completed,
        scipy_results,
        saddle_results,
        scipy_diag,
        saddle_diag,
    )

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    for i in range(
        N_REALIZATIONS
    ):

        if completed[i]:

            print(
                f"\nSkipping realization "
                f"{i + 1}/{N_REALIZATIONS} "
                "(already in checkpoint)."
            )

            continue

        seed = int(
            seeds[i]
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"REALIZATION "
            f"{i + 1}/{N_REALIZATIONS}"
        )

        print(
            "=" * 60
        )

        print(
            "Seed:",
            seed,
        )

        D_noisy = generate_noisy_dataset(
            D_clean,
            seed,
        )

        # ====================================================
        # Equal-Beta SciPy VarPro
        # ====================================================

        print(
            "\nRunning Equal-Beta SciPy VarPro..."
        )

        try:

            (
                estimate_scipy,
                result_scipy,
                diag_scipy,
            ) = run_equal_beta_scipy(
                D_noisy
            )

            store_parameter_estimate(
                scipy_results,
                i,
                estimate_scipy,
            )

            scipy_diag[
                "success"
            ][i] = True

            scipy_diag[
                "runtime"
            ][i] = diag_scipy[
                "runtime"
            ]

            scipy_diag[
                "nit"
            ][i] = diag_scipy[
                "nit"
            ]

            scipy_diag[
                "nfev"
            ][i] = diag_scipy[
                "nfev"
            ]

            scipy_diag[
                "objective"
            ][i] = diag_scipy[
                "objective"
            ]

            print(
                "Equal-Beta SciPy success."
            )

            print(
                "  theta =",
                diag_scipy[
                    "theta"
                ],
            )

            print(
                "  beta  =",
                diag_scipy[
                    "beta"
                ],
            )

            print(
                "  runtime = "
                f"{diag_scipy['runtime']:.3f} s"
            )

        except Exception as exc:

            scipy_diag[
                "success"
            ][i] = False

            print(
                "Equal-Beta SciPy FAILED:"
            )

            print(
                repr(exc)
            )

            log_error(
                i,
                seed,
                SCIPY_NAME,
                exc,
            )

        # ====================================================
        # AHU-VarPro-Saddle
        # ====================================================

        print(
            "\nRunning AHU-VarPro-Saddle..."
        )

        try:

            (
                estimate_saddle,
                solution_saddle,
                diag_saddle,
            ) = run_ahu_saddle(
                D_noisy
            )

            store_parameter_estimate(
                saddle_results,
                i,
                estimate_saddle,
            )

            saddle_diag[
                "success"
            ][i] = True

            for key in [
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
                "B_min_eig",
                "B_cond",
            ]:

                saddle_diag[
                    key
                ][i] = diag_saddle[
                    key
                ]

            print(
                "AHU-VarPro-Saddle success."
            )

            print(
                "  theta =",
                diag_saddle[
                    "theta"
                ],
            )

            print(
                "  beta1 =",
                estimate_saddle[
                    "beta1"
                ],
            )

            print(
                "  beta2 =",
                estimate_saddle[
                    "beta2"
                ],
            )

            print(
                "  stationarity =",
                diag_saddle[
                    "stationarity_norm"
                ],
            )

            print(
                "  h =",
                diag_saddle[
                    "h"
                ],
            )

            print(
                "  runtime = "
                f"{diag_saddle['runtime']:.3f} s"
            )

        except Exception as exc:

            saddle_diag[
                "success"
            ][i] = False

            print(
                "AHU-VarPro-Saddle FAILED:"
            )

            print(
                repr(exc)
            )

            log_error(
                i,
                seed,
                SADDLE_NAME,
                exc,
            )

        # ====================================================
        # Paired comparison
        # ====================================================

        if (
            scipy_diag["success"][i]
            and saddle_diag["success"][i]
        ):

            theta_scipy = np.array(
                [
                    scipy_results[
                        "T11"
                    ][i],
                    scipy_results[
                        "T12"
                    ][i],
                    scipy_results[
                        "T21"
                    ][i],
                    scipy_results[
                        "T22"
                    ][i],
                ],
                dtype=float,
            )

            theta_saddle = np.array(
                [
                    saddle_results[
                        "T11"
                    ][i],
                    saddle_results[
                        "T12"
                    ][i],
                    saddle_results[
                        "T21"
                    ][i],
                    saddle_results[
                        "T22"
                    ][i],
                ],
                dtype=float,
            )

            theta_difference = (
                theta_scipy
                - theta_saddle
            )

            print(
                "\nPaired solver comparison:"
            )

            print(
                "  Equal-Beta SciPy theta "
                "- AHU theta ="
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
                ),
            )

            beta_difference = (
                scipy_results[
                    "beta1"
                ][i]
                - saddle_results[
                    "beta1"
                ][i]
            )

            print(
                "  beta difference =",
                beta_difference,
            )

        # ====================================================
        # Mark attempted realization complete and checkpoint
        # ====================================================

        completed[i] = True

        save_checkpoint(
            completed,
            scipy_results,
            saddle_results,
            scipy_diag,
            saddle_diag,
        )

        write_csv(
            seeds,
            completed,
            scipy_results,
            saddle_results,
            scipy_diag,
            saddle_diag,
        )

        print(
            "\nCheckpoint saved."
        )

    # --------------------------------------------------------
    # Final timing
    # --------------------------------------------------------

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    scipy_successes = int(
        np.sum(
            scipy_diag[
                "success"
            ]
        )
    )

    saddle_successes = int(
        np.sum(
            saddle_diag[
                "success"
            ]
        )
    )

    paired_results, paired_mask = (
        build_paired_results(
            scipy_results,
            saddle_results,
            scipy_diag,
            saddle_diag,
        )
    )

    paired_successes = int(
        np.sum(
            paired_mask
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MONTE CARLO COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nTotal wall-clock time: "
        f"{total_runtime:.3f} s"
    )

    print(
        "\nEqual-Beta SciPy successes:",
        scipy_successes,
        "/",
        N_REALIZATIONS,
    )

    print(
        "AHU successes:",
        saddle_successes,
        "/",
        N_REALIZATIONS,
    )

    print(
        "Paired successes:",
        paired_successes,
        "/",
        N_REALIZATIONS,
    )

    # --------------------------------------------------------
    # Final CSV / checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        completed,
        scipy_results,
        saddle_results,
        scipy_diag,
        saddle_diag,
    )

    write_csv(
        seeds,
        completed,
        scipy_results,
        saddle_results,
        scipy_diag,
        saddle_diag,
    )

    # --------------------------------------------------------
    # Paired statistics
    # --------------------------------------------------------

    if paired_successes > 0:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "PAIRED MONTE CARLO STATISTICS"
        )

        print(
            "=" * 60
        )

        print_solver_summary(
            paired_results,
            TRUE_PHYSICAL_PARAMS,
            THETA0,
        )

        # ----------------------------------------------------
        # Plots
        # ----------------------------------------------------

        if GENERATE_PLOTS:

            PLOTS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            plot_all_solver_comparisons(
                results_by_solver=
                    paired_results,

                true_params=
                    TRUE_PHYSICAL_PARAMS,

                theta0=
                    THETA0,

                bins=
                    30,

                output_dir=
                    PLOTS_DIR,

                show=
                    False,

                density=
                    True,
            )

    else:

        print(
            "\nNo paired successful realizations. "
            "Plots/statistics were not generated."
        )

    # --------------------------------------------------------
    # Final paths
    # --------------------------------------------------------

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

    if GENERATE_PLOTS:
        print(
            "\nPlots:"
        )

        print(
            PLOTS_DIR
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

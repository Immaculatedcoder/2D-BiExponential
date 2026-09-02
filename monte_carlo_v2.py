import csv
from pathlib import Path
import numpy as np

from config import (
    TI, TE, TRUE_PARAMS, SIGMA, RANDOM_SEED,
    THETA0, THETA_LOWER, THETA_UPPER,
)
from model import biexponential_2d
from scipy_estimate import estimate_parameters_scipy

from saddle2 import (
    build_theta_preconditioner,
    solve_saddle_preconditioned,
    reduced_quantities,
)

from plotting_mc import (
    plot_all_solver_comparisons,
    print_solver_summary,
)

# ---------------- Settings ----------------

N_REALIZATIONS = 5

OUTPUT_DIR = Path("mc_results_slsqp_vs_ahu")
PLOTS_DIR = OUTPUT_DIR / "plots"
CSV_FILE = OUTPUT_DIR / "results.csv"

PARAMETERS = [
    "k1", "beta1", "k2", "beta2",
    "T11", "T12", "T21", "T22",
]

SLSQP_NAME = "Direct SciPy SLSQP"
AHU_NAME = "AHU-VarPro-Saddle"

BETA0 = 1.0
EPS_K = 1e-8

MU0 = np.zeros(6)
LAM0 = 0.0
T_SPAN = (0.0, 10000.0)

P_THETA = build_theta_preconditioner(
    THETA_LOWER,
    THETA_UPPER,
)
P_MU = np.eye(6)
P_LAM = 1.0

GAMMA_THETA = 1.0
GAMMA_MU = 1.0
GAMMA_LAM = 1.0

AHU_METHOD = "BDF"
AHU_RTOL = 1e-6
AHU_ATOL = 1e-8
AHU_MAX_STEP = 20.0
AHU_STEADY_TOL = 1e-5
B_EIG_TOL = 1e-8


def empty_results():
    return {
        name: np.full(N_REALIZATIONS, np.nan)
        for name in PARAMETERS
    }


def store(results, i, estimate):
    for name in PARAMETERS:
        results[name][i] = estimate[name]


def run_slsqp(D_noisy):
    estimate, diagnostics, result = estimate_parameters_scipy(
        D_noisy,
        TI,
        TE,
        THETA0,
        THETA_LOWER,
        THETA_UPPER,
        beta0=BETA0,
    )

    return estimate, {
        "objective": float(diagnostics["objective"]),
        "h": float(diagnostics["h"]),
    }


def run_ahu(D_noisy):
    d = np.asarray(
        D_noisy,
        dtype=float,
    ).ravel(order="F")

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
        method=AHU_METHOD,
        rtol=AHU_RTOL,
        atol=AHU_ATOL,
        max_step=AHU_MAX_STEP,
        b_eig_tol=B_EIG_TOL,
        steady_tol=AHU_STEADY_TOL,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    y = solution.y[:, -1]
    theta = y[:4]
    mu = y[4:10]
    lam = float(y[10])

    c, residual, g, h = reduced_quantities(
        TI,
        TE,
        theta,
        mu,
        lam,
        d,
        EPS_K,
    )

    k1, k1_hat, k2, k2_hat = c

    estimate = {
        "k1": float(k1),
        "beta1": float(k1_hat / k1),
        "k2": float(k2),
        "beta2": float(k2_hat / k2),
        "T11": float(theta[0]),
        "T12": float(theta[1]),
        "T21": float(theta[2]),
        "T22": float(theta[3]),
    }

    return estimate, {
        "objective": 0.5 * float(residual @ residual),
        "h": float(h),
        "final_time": float(solution.t[-1]),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    D_clean = biexponential_2d(
        TI,
        TE,
        TRUE_PARAMS,
    )

    slsqp_results = empty_results()
    ahu_results = empty_results()

    slsqp_success = np.zeros(
        N_REALIZATIONS,
        dtype=bool,
    )
    ahu_success = np.zeros(
        N_REALIZATIONS,
        dtype=bool,
    )

    rows = []

    for i in range(N_REALIZATIONS):
        seed = int(RANDOM_SEED) + i
        rng = np.random.default_rng(seed)

        D_noisy = D_clean + rng.normal(
            0.0,
            SIGMA,
            size=D_clean.shape,
        )

        print(
            f"\nRealization {i+1}/{N_REALIZATIONS} "
            f"(seed={seed})"
        )

        try:
            est_slsqp, diag_slsqp = run_slsqp(
                D_noisy
            )
            store(
                slsqp_results,
                i,
                est_slsqp,
            )
            slsqp_success[i] = True

            print(
                "  SLSQP:",
                f"F={diag_slsqp['objective']:.8f}",
                f"h={diag_slsqp['h']:.2e}",
            )
        except Exception as exc:
            est_slsqp = None
            print(
                "  SLSQP FAILED:",
                repr(exc),
            )

        try:
            est_ahu, diag_ahu = run_ahu(
                D_noisy
            )
            store(
                ahu_results,
                i,
                est_ahu,
            )
            ahu_success[i] = True

            print(
                "  AHU  :",
                f"F={diag_ahu['objective']:.8f}",
                f"h={diag_ahu['h']:.2e}",
                f"t={diag_ahu['final_time']:.2f}",
            )
        except Exception as exc:
            est_ahu = None
            print(
                "  AHU FAILED:",
                repr(exc),
            )

        row = {
            "realization": i + 1,
            "seed": seed,
            "slsqp_success": slsqp_success[i],
            "ahu_success": ahu_success[i],
        }

        for name in PARAMETERS:
            row[f"slsqp_{name}"] = (
                np.nan
                if est_slsqp is None
                else est_slsqp[name]
            )
            row[f"ahu_{name}"] = (
                np.nan
                if est_ahu is None
                else est_ahu[name]
            )

        rows.append(row)

    paired = (
        slsqp_success
        & ahu_success
    )

    results_by_solver = {
        SLSQP_NAME: {
            name: slsqp_results[name][paired]
            for name in PARAMETERS
        },
        AHU_NAME: {
            name: ahu_results[name][paired]
            for name in PARAMETERS
        },
    }

    true_params = {
        name: float(TRUE_PARAMS[name])
        for name in PARAMETERS
    }

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 55)
    print("MONTE CARLO COMPLETE")
    print("=" * 55)
    print(
        "SLSQP successes:",
        int(slsqp_success.sum()),
        "/",
        N_REALIZATIONS,
    )
    print(
        "AHU successes:",
        int(ahu_success.sum()),
        "/",
        N_REALIZATIONS,
    )
    print(
        "Paired successes:",
        int(paired.sum()),
        "/",
        N_REALIZATIONS,
    )

    if paired.any():
        print_solver_summary(
            results_by_solver,
            true_params,
            THETA0,
        )

        PLOTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_all_solver_comparisons(
            results_by_solver=results_by_solver,
            true_params=true_params,
            theta0=THETA0,
            bins=30,
            output_dir=PLOTS_DIR,
            show=False,
            density=True,
        )

    print("\nCSV:", CSV_FILE)
    print("Plots:", PLOTS_DIR)


if __name__ == "__main__":
    main()

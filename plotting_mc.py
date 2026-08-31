import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Parameter information
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


PARAMETER_LABELS = {
    "k1": r"$k_1$",
    "beta1": r"$\beta_1$",
    "k2": r"$k_2$",
    "beta2": r"$\beta_2$",
    "T11": r"$T_{11}$",
    "T12": r"$T_{12}$",
    "T21": r"$T_{21}$",
    "T22": r"$T_{22}$",
}


THETA_NAMES = [
    "T11",
    "T12",
    "T21",
    "T22",
]


# ============================================================
# Helper: format theta0 for figure titles
# ============================================================

def format_theta0(theta0):
    """
    Convert theta0 into a compact string.

    Example:

        theta0 = [980, 2250, 75, 260]

    becomes

        T11=980, T12=2250, T21=75, T22=260
    """

    theta0 = np.asarray(
        theta0,
        dtype=float
    )

    if theta0.shape != (4,):
        raise ValueError(
            "theta0 must have shape (4,)."
        )

    pieces = []

    for name, value in zip(
        THETA_NAMES,
        theta0
    ):
        pieces.append(
            f"{name}={value:g}"
        )

    return ", ".join(pieces)


# ============================================================
# Helper: clean result values
# ============================================================

def clean_values(values):
    """
    Remove NaN and infinite values.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    return values[
        np.isfinite(values)
    ]


# ============================================================
# Plot distribution for ONE parameter and TWO OR MORE solvers
# ============================================================

def plot_parameter_comparison(
    results_by_solver,
    parameter_name,
    true_value,
    theta0,
    bins=30,
    output_dir=None,
    show=True,
    density=True
):
    """
    Compare Monte Carlo distributions from multiple solvers
    on the same figure.

    Parameters
    ----------
    results_by_solver : dict

        Example:

        {
            "SciPy VarPro": {
                "T11": [...],
                ...
            },

            "Preconditioned Saddle": {
                "T11": [...],
                ...
            }
        }

    parameter_name : str
        Parameter to plot.

    true_value : float
        Ground-truth value.

    theta0 : ndarray, shape (4,)
        Initial nonlinear parameter vector.

    bins : int
        Number of common histogram bins.

    output_dir : str or Path or None
        Folder where figure should be saved.

    show : bool
        Whether to display the figure.

    density : bool
        If True, plot probability density rather than raw
        frequency counts.

        Density is preferred when comparing solvers because
        the number of successful realizations may differ.
    """

    if parameter_name not in PARAMETER_NAMES:
        raise ValueError(
            f"Unknown parameter: {parameter_name}"
        )

    if len(results_by_solver) == 0:
        raise ValueError(
            "results_by_solver cannot be empty."
        )

    label = PARAMETER_LABELS.get(
        parameter_name,
        parameter_name
    )

    # --------------------------------------------------------
    # Collect finite results from every solver
    # --------------------------------------------------------

    cleaned = {}

    all_values = []

    for solver_name, results in results_by_solver.items():

        if parameter_name not in results:
            raise KeyError(
                f"{solver_name} is missing "
                f"parameter {parameter_name}."
            )

        values = clean_values(
            results[parameter_name]
        )

        if values.size == 0:
            raise ValueError(
                f"No finite {parameter_name} estimates "
                f"for solver {solver_name}."
            )

        cleaned[solver_name] = values

        all_values.append(
            values
        )

    # --------------------------------------------------------
    # Use exactly the SAME bins for every solver
    #
    # This is very important for a fair visual comparison.
    # --------------------------------------------------------

    combined_values = np.concatenate(
        all_values
    )

    bin_edges = np.histogram_bin_edges(
        combined_values,
        bins=bins
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # --------------------------------------------------------
    # Plot each solver distribution
    # --------------------------------------------------------

    for solver_name, values in cleaned.items():

        ax.hist(
            values,
            bins=bin_edges,
            density=density,
            alpha=0.45,
            edgecolor="black",
            label=(
                f"{solver_name} "
                f"(N={values.size})"
            )
        )

    # --------------------------------------------------------
    # True parameter
    # --------------------------------------------------------

    ax.axvline(
        true_value,
        linewidth=2.5,
        linestyle="-",
        label=(
            f"True {parameter_name} "
            f"= {true_value:.8g}"
        )
    )

    # --------------------------------------------------------
    # Mean from every solver
    # --------------------------------------------------------

    for solver_name, values in cleaned.items():

        mean_value = np.mean(
            values
        )

        ax.axvline(
            mean_value,
            linewidth=1.8,
            linestyle="--",
            label=(
                f"{solver_name} mean "
                f"= {mean_value:.8g}"
            )
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    theta0_text = format_theta0(
        theta0
    )

    solver_text = " vs ".join(
        results_by_solver.keys()
    )

    title = (
        f"Monte Carlo Distribution of {label}\n"
        f"{solver_text}\n"
        f"True {parameter_name} = {true_value:g} | "
        f"Initial theta: {theta0_text}"
    )

    ax.set_title(
        title,
        fontsize=11
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        f"Estimated {label}"
    )

    if density:
        ax.set_ylabel(
            "Probability Density"
        )
    else:
        ax.set_ylabel(
            "Frequency"
        )

    ax.legend(
        fontsize=8
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if output_dir is not None:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_path = (
            output_dir
            / f"{parameter_name}_solver_comparison.png"
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    else:
        plt.close(fig)

    return fig, ax


# ============================================================
# Plot ALL parameter comparisons
# ============================================================

def plot_all_solver_comparisons(
    results_by_solver,
    true_params,
    theta0,
    bins=30,
    output_dir=None,
    show=True,
    density=True
):
    """
    Generate one solver-comparison distribution plot
    for every physical parameter.
    """

    for parameter_name in PARAMETER_NAMES:

        if parameter_name not in true_params:
            raise KeyError(
                f"true_params is missing "
                f"{parameter_name}."
            )

        plot_parameter_comparison(
            results_by_solver=results_by_solver,
            parameter_name=parameter_name,
            true_value=true_params[parameter_name],
            theta0=theta0,
            bins=bins,
            output_dir=output_dir,
            show=show,
            density=density
        )


# ============================================================
# Statistics for ONE parameter
# ============================================================

def summarize_parameter_distribution(
    values,
    true_value
):
    """
    Monte Carlo statistics for one estimated parameter.

    Returns

        N
        true
        mean
        std
        bias
        relative_bias
        percent_bias
        rmse
        min
        max
    """

    values = clean_values(
        values
    )

    if values.size == 0:
        raise ValueError(
            "No finite estimates available."
        )

    mean = np.mean(
        values
    )

    if values.size > 1:

        std = np.std(
            values,
            ddof=1
        )

    else:

        std = 0.0

    bias = (
        mean
        - true_value
    )

    if true_value != 0.0:

        relative_bias = (
            bias
            / true_value
        )

        percent_bias = (
            100.0
            * relative_bias
        )

    else:

        relative_bias = np.nan
        percent_bias = np.nan

    rmse = np.sqrt(
        np.mean(
            (values - true_value) ** 2
        )
    )

    return {

        "N":
            values.size,

        "true":
            true_value,

        "mean":
            mean,

        "std":
            std,

        "bias":
            bias,

        "relative_bias":
            relative_bias,

        "percent_bias":
            percent_bias,

        "rmse":
            rmse,

        "min":
            np.min(values),

        "max":
            np.max(values),
    }


# ============================================================
# Statistics for EVERY parameter of ONE solver
# ============================================================

def summarize_solver(
    results,
    true_params
):
    """
    Summarize all parameters for one solver.
    """

    summary = {}

    for parameter_name in PARAMETER_NAMES:

        if parameter_name not in results:
            raise KeyError(
                f"Missing Monte Carlo result: "
                f"{parameter_name}"
            )

        if parameter_name not in true_params:
            raise KeyError(
                f"Missing true parameter: "
                f"{parameter_name}"
            )

        summary[parameter_name] = (
            summarize_parameter_distribution(
                results[parameter_name],
                true_params[parameter_name]
            )
        )

    return summary


# ============================================================
# Statistics for ALL solvers
# ============================================================

def summarize_all_solvers(
    results_by_solver,
    true_params
):
    """
    Produce Monte Carlo summaries for every solver.

    Output:

        summary["SciPy VarPro"]["T11"]
        summary["Preconditioned Saddle"]["T11"]
        ...
    """

    summary = {}

    for solver_name, results in results_by_solver.items():

        summary[solver_name] = (
            summarize_solver(
                results,
                true_params
            )
        )

    return summary


# ============================================================
# Print solver comparison table
# ============================================================

def print_solver_summary(
    results_by_solver,
    true_params,
    theta0
):
    """
    Print mean, standard deviation, bias and RMSE
    for all parameters and all solvers.
    """

    summary = summarize_all_solvers(
        results_by_solver,
        true_params
    )

    print(
        "\n============================================================"
    )

    print(
        "MONTE CARLO SOLVER COMPARISON"
    )

    print(
        "============================================================"
    )

    print(
        "\nInitial theta:"
    )

    print(
        np.asarray(theta0)
    )

    print(
        "\nTrue parameters:"
    )

    for name in PARAMETER_NAMES:

        print(
            f"{name:6s} = "
            f"{true_params[name]:.8g}"
        )

    for solver_name, solver_summary in summary.items():

        print(
            "\n============================================================"
        )

        print(
            solver_name
        )

        print(
            "============================================================"
        )

        print(
            f"{'Parameter':<10}"
            f"{'True':>14}"
            f"{'Mean':>14}"
            f"{'Std':>14}"
            f"{'Bias':>14}"
            f"{'RMSE':>14}"
        )

        print(
            "-" * 80
        )

        for parameter_name in PARAMETER_NAMES:

            stats = solver_summary[
                parameter_name
            ]

            print(
                f"{parameter_name:<10}"
                f"{stats['true']:>14.6g}"
                f"{stats['mean']:>14.6g}"
                f"{stats['std']:>14.6g}"
                f"{stats['bias']:>14.6g}"
                f"{stats['rmse']:>14.6g}"
            )
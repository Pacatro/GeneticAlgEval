import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

type ParamsType = dict[str, int | float]

JAVA_MAIN = [
    "java",
    "-cp",
    "target/classes",
    "es.uma.informatica.misia.ae.simpleea.Main",
]

EXECUTIONS = 30
POPULATION_SIZE = 100
FUNCTION_EVALUATIONS = 3000  # If 0 -> stop at optimal solution
P_MUT = [round(x * 0.01, 2) for x in range(1, 11)]
P_CROSS = [round(x * 0.1, 1) for x in range(1, 11)]
PROBLEM_SIZES = [50, 100]
EVAL_PARAMS: list[ParamsType] = [
    {
        "population size": POPULATION_SIZE,
        "function evaluations": FUNCTION_EVALUATIONS,
        "bitflip probability": p_mut,
        "cross probability": p_cross,
    }
    for p_mut in P_MUT
    for p_cross in P_CROSS
]


def run_java(
    params: ParamsType, problem_size: int, function_evaluations: int
) -> tuple[float, float]:
    """Execute the Java evolutionary algorithm with given parameters."""
    cmd = JAVA_MAIN + [
        str(params["population size"]),
        str(function_evaluations),
        str(params["bitflip probability"]),
        str(params["cross probability"]),
        str(problem_size),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error running java command: {result.stderr}")

    java_out = result.stdout.strip().split(",")
    fitness_result = float(java_out[0])
    elapsed_time = float(java_out[1])

    return fitness_result, elapsed_time


def eval_results(problem_size: int, function_evaluations: int) -> pd.DataFrame:
    """Run all parameter combinations and collect results."""
    results = []

    for params in EVAL_PARAMS:
        for execution in range(EXECUTIONS):
            result, elapsed_time = run_java(params, problem_size, function_evaluations)

            result_column = "Fitness" if function_evaluations != 0 else "Evaluations"

            result_row = {
                "Execution": execution,
                "Population Size": params["population size"],
                "Problem Size": problem_size,
                "Evaluations": function_evaluations,
                "Mutation Probability": params["bitflip probability"],
                "Crossover Probability": params["cross probability"],
                result_column: result,
                "Elapsed Time (ms)": elapsed_time,
            }

            if function_evaluations != 0:
                result_row["Max Evaluations"] = function_evaluations

            results.append(result_row)

    return pd.DataFrame(results)


def aggregate_stats(
    df: pd.DataFrame, group_cols: list[str], value_col: str
) -> pd.DataFrame:
    """Aggregate mean and std statistics for given grouping columns."""
    result_stats = df.groupby(group_cols)[value_col].agg(["mean", "std"])
    result_stats.columns = [f"{value_col}_mean", f"{value_col}_std"]

    time_stats = df.groupby(group_cols)["Elapsed Time (ms)"].agg(["mean", "std"])
    time_stats.columns = ["time_mean", "time_std"]

    assert isinstance(result_stats, pd.DataFrame)
    assert isinstance(time_stats, pd.DataFrame)
    return pd.concat([result_stats, time_stats], axis=1)


def generate_stats(
    df: pd.DataFrame, result_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate statistics grouped by different parameter combinations."""
    mix_stats = aggregate_stats(
        df, ["Mutation Probability", "Crossover Probability"], result_column
    )
    cross_stats = aggregate_stats(df, ["Crossover Probability"], result_column)
    mut_stats = aggregate_stats(df, ["Mutation Probability"], result_column)

    return mix_stats, cross_stats, mut_stats


def plot_single_parameter(
    stats_path: str,
    xlabel: str,
    ylabel: str,
    img_path: str | None = None,
    group_param: str | None = None,
    param_x_name: str | None = None,
    param_y_name: str | None = None,
) -> None:
    """Create a line plot for parameter analysis."""
    stats = pd.read_csv(stats_path)
    param_x_name = param_x_name if param_x_name else xlabel
    param_y_name = param_y_name if param_y_name else ylabel

    plt.figure(figsize=(8, 6))

    if group_param:
        for hue_value in stats[group_param].unique():
            data = stats[stats[group_param] == hue_value]
            plt.plot(
                data[param_x_name],
                data[f"{param_y_name}_mean"],
                marker="o",
                linewidth=2,
                markersize=8,
                label=f"{hue_value}",
            )
    else:
        plt.plot(
            stats[param_x_name],
            stats[f"{param_y_name}_mean"],
            marker="o",
            linewidth=2,
            markersize=8,
        )

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    if group_param:
        plt.legend(title=group_param, loc="best")

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if img_path:
        print(f"Saving plot to {img_path}")
        plt.savefig(img_path, dpi=200, bbox_inches="tight")
        plt.close()


def plot_heatmap(
    size: int,
    ylabel: str,
    evals_folder_name: str,
    img_path: str | None = None,
    param_y_name: str | None = None,
) -> None:
    """Create a heatmap of mutation vs crossover probabilities."""
    df = pd.read_csv(f"./{evals_folder_name}/size={size}/results.csv")
    param_y_name = param_y_name if param_y_name else ylabel

    pivot_data = (
        df.groupby(["Mutation Probability", "Crossover Probability"])[param_y_name]
        .mean()
        .reset_index()
    )

    pivot_table = pivot_data.pivot(
        index="Mutation Probability",
        columns="Crossover Probability",
        values=param_y_name,
    )

    plt.figure(figsize=(12, 8))
    cmap = "YlOrRd" if ylabel == "Fitness" else "YlGnBu_r"

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        cmap=cmap,
        cbar_kws={"label": ylabel},
        linewidths=0.5,
    )

    cbar = ax.collections[0].colorbar
    assert cbar is not None
    cbar.set_label(ylabel, fontsize=20)

    plt.xlabel("P_c", fontsize=20)
    plt.ylabel("P_m", fontsize=20)
    plt.tight_layout()

    if img_path:
        print(f"Saving heatmap to {img_path}")
        plt.savefig(img_path, dpi=200, bbox_inches="tight")
        plt.close()


def generate_all_plots(
    problem_size: int,
    size_folder: Path,
    result_column: str,
    ylabel: str,
) -> None:
    """Generate all visualization plots for a given problem size."""
    stats_filename = size_folder / "stats"
    result_filename = size_folder / result_column

    # Plot by P_c only
    plot_single_parameter(
        f"{stats_filename}_cross.csv",
        xlabel="P_c",
        ylabel=ylabel,
        img_path=f"{result_filename}_by_cross.png",
        param_x_name="Crossover Probability",
        param_y_name=result_column,
    )

    # Plot by P_m only
    plot_single_parameter(
        f"{stats_filename}_mut.csv",
        xlabel="P_m",
        ylabel=ylabel,
        img_path=f"{result_filename}_by_mut.png",
        param_x_name="Mutation Probability",
        param_y_name=result_column,
    )

    # Plot by P_m grouped by P_c
    plot_single_parameter(
        f"{stats_filename}_mix.csv",
        xlabel="P_m",
        ylabel=ylabel,
        img_path=f"{result_filename}_by_mix_mut.png",
        group_param="Crossover Probability",
        param_x_name="Mutation Probability",
        param_y_name=result_column,
    )

    # Plot by P_c grouped by P_m
    plot_single_parameter(
        f"{stats_filename}_mix.csv",
        xlabel="P_c",
        ylabel=ylabel,
        img_path=f"{result_filename}_by_mix_cross.png",
        group_param="Mutation Probability",
        param_x_name="Crossover Probability",
        param_y_name=result_column,
    )

    plot_heatmap(
        size=problem_size,
        ylabel=ylabel,
        evals_folder_name=size_folder.parent.name,
        img_path=f"{result_filename}_heatmap.png",
        param_y_name=result_column,
    )


def run_evaluation_pipeline(
    problem_size: int,
    function_evaluations: int,
    result_column: str,
    size_folder: Path,
) -> None:
    """Execute evaluation pipeline: run experiments and generate statistics."""
    print(f"Starting evaluations for problem size {problem_size}")

    df = eval_results(problem_size, function_evaluations)
    df.to_csv(size_folder / "results.csv", index=False)

    print(f"Generating statistics for problem size {problem_size}")

    mix_stats, cross_stats, mut_stats = generate_stats(df, result_column)

    stats_filename = size_folder / "stats"
    mix_stats.to_csv(f"{stats_filename}_mix.csv")
    cross_stats.to_csv(f"{stats_filename}_cross.csv")
    mut_stats.to_csv(f"{stats_filename}_mut.csv")


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate evolutionary algorithm with different parameters"
    )
    parser.add_argument(
        "--global-optimum",
        "-g",
        action="store_true",
        help="Stop algorithm when global optimum is found",
    )
    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Run evaluations (otherwise only generate plots)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    function_evaluations = 0 if args.global_optimum else FUNCTION_EVALUATIONS
    evals_folder_name = "evals_global" if args.global_optimum else "evals_max_iter"
    result_column = "Evaluations" if args.global_optimum else "Fitness"
    ylabel = "Evaluaciones" if args.global_optimum else "Fitness"

    evals_folder = Path(evals_folder_name)
    evals_folder.mkdir(exist_ok=True, parents=True)

    for problem_size in PROBLEM_SIZES:
        size_folder = evals_folder / f"size={problem_size}"
        size_folder.mkdir(exist_ok=True, parents=True)

        if args.eval:
            run_evaluation_pipeline(
                problem_size, function_evaluations, result_column, size_folder
            )
        else:
            generate_all_plots(problem_size, size_folder, result_column, ylabel)


if __name__ == "__main__":
    main()

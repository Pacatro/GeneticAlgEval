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
FUNCTION_EVALUATIONS = 5000  # If 0 -> stop at optimal solution
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
    cmd = JAVA_MAIN + [
        str(params["population size"]),
        str(function_evaluations),
        str(params["bitflip probability"]),
        str(params["cross probability"]),
        str(problem_size),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Error running java command: {result.stderr}")

    java_out = result.stdout.strip().split(",")
    result = float(java_out[0])
    elapsed_time = float(java_out[1])
    return result, elapsed_time


def eval_results(problem_size: int, function_evaluations: int) -> pd.DataFrame:
    results = []

    for params in EVAL_PARAMS:
        for execution in range(EXECUTIONS):
            result, elapsed_time = run_java(params, problem_size, function_evaluations)
            result_row = {}
            result_row["Execution"] = execution
            result_row["Population Size"] = params["population size"]
            result_row["Problem Size"] = problem_size
            result_row["Evaluations"] = function_evaluations
            result_row["Mutation Probability"] = params["bitflip probability"]
            result_row["Crossover Probability"] = params["cross probability"]
            result_row["Fitness" if function_evaluations != 0 else "Evaluations"] = (
                result
            )

            if function_evaluations != 0:
                result_row["Max Evaluations"] = function_evaluations

            result_row["Elapsed Time (ms)"] = elapsed_time
            results.append(result_row)

    return pd.DataFrame(results)


def generate_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mix_stats = df.groupby(["Mutation Probability", "Crossover Probability"])[
        "Fitness"
    ].agg(["mean", "std"])
    cross_stats = df.groupby(["Crossover Probability"])["Fitness"].agg(["mean", "std"])
    mut_stats = df.groupby(["Mutation Probability"])["Fitness"].agg(["mean", "std"])

    assert isinstance(mix_stats, pd.DataFrame)
    assert isinstance(cross_stats, pd.DataFrame)
    assert isinstance(mut_stats, pd.DataFrame)
    return mix_stats, cross_stats, mut_stats


def plot_single_parameter(
    stats: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    img_path: str | None = None,
    group_param: str | None = None,
    param_x_name: str | None = None,
) -> None:
    stats_reset = stats.reset_index()
    param_x_name = param_x_name if param_x_name else xlabel

    plt.figure(figsize=(12, 6))

    if group_param:
        for hue_value in stats_reset[group_param].unique():
            data = stats_reset[stats_reset[group_param] == hue_value]
            plt.plot(
                data[param_x_name],
                data["mean"],
                marker="o",
                linewidth=2,
                markersize=8,
                label=f"{hue_value}",
            )
    else:
        plt.plot(
            stats_reset[param_x_name],
            stats_reset["mean"],
            marker="o",
            linewidth=2,
            markersize=8,
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{title}", fontsize=14)

    if group_param:
        plt.legend(title=group_param, loc="best")

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if img_path:
        print(f"Saving plot to {img_path}")
        plt.savefig(img_path, dpi=200, bbox_inches="tight")


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    ylabel: str = "Fitness",
    img_path: str | None = None,
) -> None:
    pivot_data = (
        df.groupby(["Mutation Probability", "Crossover Probability"])["Fitness"]
        .mean()
        .reset_index()
    )

    pivot_table = pivot_data.pivot(
        index="Mutation Probability", columns="Crossover Probability", values="Fitness"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd" if ylabel == "Fitness" else "YlGnBu_r",
        cbar_kws={"label": ylabel},
        linewidths=0.5,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Crossover Probability", fontsize=12)
    plt.ylabel("Mutation Probability", fontsize=12)
    plt.tight_layout()

    if img_path:
        print(f"Saving heatmap to {img_path}")
        plt.savefig(img_path, dpi=200, bbox_inches="tight")


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-optimum",
        "-g",
        action="store_true",
        help="Stop algorithm when global optimum is found",
    )
    return parser


def main():
    args = get_args().parse_args()

    function_evaluations = FUNCTION_EVALUATIONS if not args.global_optimum else 0
    evals_folder_name = "evals_global" if args.global_optimum else "evals_max_iter"

    evals_folder = Path(evals_folder_name)
    evals_folder.mkdir(exist_ok=True, parents=True)

    for problem_size in PROBLEM_SIZES:
        size_eval_folder = evals_folder / f"size={problem_size}"
        size_eval_folder.mkdir(exist_ok=True, parents=True)
        results_filename = size_eval_folder / "results"
        results_filename = size_eval_folder / "stats"

        print("Starting evaluations for problem size", problem_size)

        df = eval_results(problem_size, function_evaluations)
        df.to_csv(f"{results_filename}.csv")

        print("Generating statistics for problem size", problem_size)
        mix_stats, cross_stats, mut_stats = generate_stats(df)

        mix_stats.to_csv(f"{results_filename}_mix.csv")
        cross_stats.to_csv(f"{results_filename}_cross.csv")
        mut_stats.to_csv(f"{results_filename}_mut.csv")

        y_label = "Evaluations" if args.global_optimum else "Fitness"
        sigle_param_filename = size_eval_folder / y_label

        plot_single_parameter(
            cross_stats,
            title=f"{y_label} evolution for different crossover probabilities (Size={problem_size})",
            xlabel="Crossover Probability",
            ylabel=y_label,
            img_path=f"{sigle_param_filename}_by_cross.png",
        )

        plot_single_parameter(
            mut_stats,
            title=f"{y_label} evolution for different mutation probabilities (Size={problem_size})",
            xlabel="Mutation Probability",
            ylabel=y_label,
            img_path=f"{sigle_param_filename}_by_mut.png",
        )

        plot_single_parameter(
            mix_stats,
            title=f"{y_label} evolution for different configurations by Mutation (Size={problem_size})",
            xlabel="Mutation Probability",
            ylabel=y_label,
            img_path=f"{sigle_param_filename}_by_mix_mut.png",
            group_param="Crossover Probability",
        )

        plot_single_parameter(
            mix_stats,
            title=f"{y_label} evolution for different configurations by Crossover (Size={problem_size})",
            xlabel="Crossover Probability",
            ylabel=y_label,
            img_path=f"{sigle_param_filename}_by_mix_cross.png",
            group_param="Mutation Probability",
        )

        plot_heatmap(
            df,
            title=f"Heatmap for different configurations (Size={problem_size})",
            img_path=f"{sigle_param_filename}_heatmap.png",
            ylabel=y_label,
        )


if __name__ == "__main__":
    main()

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


def generate_stats(
    df: pd.DataFrame, result_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mix_stats = df.groupby(["Mutation Probability", "Crossover Probability"])[
        result_column
    ].agg(["mean", "std"])
    mix_stats.columns = [f"{result_column}_mean", f"{result_column}_std"]

    cross_stats = df.groupby(["Crossover Probability"])[result_column].agg(
        ["mean", "std"]
    )
    cross_stats.columns = [f"{result_column}_mean", f"{result_column}_std"]

    mut_stats = df.groupby(["Mutation Probability"])[result_column].agg(["mean", "std"])
    mut_stats.columns = [f"{result_column}_mean", f"{result_column}_std"]

    time_stats = df.groupby(["Mutation Probability", "Crossover Probability"])[
        "Elapsed Time (ms)"
    ].agg(["mean", "std"])
    time_stats.columns = ["time_mean", "time_std"]

    time_cross_stats = df.groupby(["Crossover Probability"])["Elapsed Time (ms)"].agg(
        ["mean", "std"]
    )
    time_cross_stats.columns = ["time_mean", "time_std"]

    time_mut_stats = df.groupby(["Mutation Probability"])["Elapsed Time (ms)"].agg(
        ["mean", "std"]
    )
    time_mut_stats.columns = ["time_mean", "time_std"]

    mix_stats = pd.concat([mix_stats, time_stats], axis=1)  # type: ignore
    cross_stats = pd.concat([cross_stats, time_cross_stats], axis=1)  # type: ignore
    mut_stats = pd.concat([mut_stats, time_mut_stats], axis=1)  # type: ignore

    return mix_stats, cross_stats, mut_stats


def plot_single_parameter(
    stats: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    img_path: str | None = None,
    group_param: str | None = None,
    param_x_name: str | None = None,
) -> None:
    stats_reset = stats.reset_index()
    param_x_name = param_x_name if param_x_name else xlabel

    plt.figure(figsize=(8, 6))

    if group_param:
        for hue_value in stats_reset[group_param].unique():
            data = stats_reset[stats_reset[group_param] == hue_value]
            plt.plot(
                data[param_x_name],
                data[f"{ylabel}_mean"],
                marker="o",
                linewidth=2,
                markersize=8,
                label=f"{hue_value}",
            )
    else:
        plt.plot(
            stats_reset[param_x_name],
            stats_reset[f"{ylabel}_mean"],
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


def plot_heatmap(
    size: int,
    ylabel: str,
    evals_folder_name: str,
    img_path: str | None = None,
) -> None:
    df = pd.read_csv(f"./{evals_folder_name}/size={size}/results.csv")

    pivot_data = (
        df.groupby(["Mutation Probability", "Crossover Probability"])[ylabel]
        .mean()
        .reset_index()
    )

    pivot_table = pivot_data.pivot(
        index="Mutation Probability", columns="Crossover Probability", values=ylabel
    )

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        cmap="YlOrRd" if ylabel == "Fitness" else "YlGnBu_r",
        cbar_kws={"label": ylabel},
        linewidths=0.5,
    )
    cbar = ax.collections[0].colorbar  # obtener el objeto Colorbar
    assert cbar is not None
    cbar.set_label(ylabel, fontsize=20)
    plt.xlabel("Crossover Probability", fontsize=20)
    plt.ylabel("Mutation Probability", fontsize=20)
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
    result_column = "Evaluations" if args.global_optimum else "Fitness"

    evals_folder = Path(evals_folder_name)
    evals_folder.mkdir(exist_ok=True, parents=True)

    for problem_size in PROBLEM_SIZES:
        size_eval_folder = evals_folder / f"size={problem_size}"
        size_eval_folder.mkdir(exist_ok=True, parents=True)
        results_filename = size_eval_folder / "results"
        stats_filename = size_eval_folder / "stats"

        # print("Starting evaluations for problem size", problem_size)
        #
        # df = eval_results(problem_size, function_evaluations)
        # df.to_csv(f"{results_filename}.csv")

        # print("Generating statistics for problem size", problem_size)
        #
        # mix_stats, cross_stats, mut_stats = generate_stats(df, result_column)
        #
        # mix_stats.to_csv(f"{stats_filename}_mix.csv")
        # cross_stats.to_csv(f"{stats_filename}_cross.csv")
        # mut_stats.to_csv(f"{stats_filename}_mut.csv")

        sigle_param_filename = size_eval_folder / result_column

        # plot_single_parameter(
        #     cross_stats,
        #     xlabel="Crossover Probability",
        #     ylabel=result_column,
        #     img_path=f"{sigle_param_filename}_by_cross.png",
        # )
        #
        # plot_single_parameter(
        #     mut_stats,
        #     xlabel="Mutation Probability",
        #     ylabel=result_column,
        #     img_path=f"{sigle_param_filename}_by_mut.png",
        # )
        #
        # plot_single_parameter(
        #     mix_stats,
        #     xlabel="Mutation Probability",
        #     ylabel=result_column,
        #     img_path=f"{sigle_param_filename}_by_mix_mut.png",
        #     group_param="Crossover Probability",
        # )
        #
        # plot_single_parameter(
        #     mix_stats,
        #     xlabel="Crossover Probability",
        #     ylabel=result_column,
        #     img_path=f"{sigle_param_filename}_by_mix_cross.png",
        #     group_param="Mutation Probability",
        # )

        plot_heatmap(
            size=problem_size,
            ylabel=result_column,
            evals_folder_name=evals_folder_name,
            img_path=f"{sigle_param_filename}_heatmap.png",
        )


if __name__ == "__main__":
    main()

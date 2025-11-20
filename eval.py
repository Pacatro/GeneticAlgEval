import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

type ParamsType = dict[str, int | float]

JAVA_MAIN = [
    "java",
    "-cp",
    "target/classes",
    "es.uma.informatica.misia.ae.simpleea.Main",
]

EXECUTIONS = 30
POPULATION_SIZE = 100
FUNCTION_EVALUATIONS = 1000  # If 0 -> stop at optimal solution
P_MUT = [0.01, 0.005, 0.001, 0.0005, 0.0001]
P_CROSS = [0.6, 0.7, 0.8, 0.9, 0.95]
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


def run_java(params: ParamsType, problem_size: int, function_evaluations: int) -> float:
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

    fitness = float(result.stdout.strip())
    return fitness


def eval_results(problem_size: int, function_evaluations: int) -> pd.DataFrame:
    results = []

    for params in EVAL_PARAMS:
        for execution in range(EXECUTIONS):
            fitness = run_java(params, problem_size, function_evaluations)
            result_row = params.copy()
            result_row["fitness"] = fitness
            result_row["execution"] = execution
            results.append(result_row)

    return pd.DataFrame(results)


def plot_results(
    df: pd.DataFrame,
    title: str,
    x_col: str = "execution",
    y_col: str = "fitness",
    img_path: str | None = None,
) -> None:
    assert x_col in df.columns, f"Columna {x_col} no encontrada"
    assert y_col in df.columns, f"Columna {y_col} no encontrada"

    plt.figure(figsize=(12, 6))
    for (p_mut, p_cross), group in df.groupby(  # type: ignore
        ["bitflip probability", "cross probability"]
    ):
        plt.plot(
            group[x_col],
            group[y_col],
            label=f"Mut={p_mut}, Cross={p_cross}",
        )
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if img_path:
        print("Saving plot to", img_path)
        plt.savefig(img_path, dpi=200, bbox_inches="tight")


def generate_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(["bitflip probability", "cross probability"])["fitness"].agg(
        ["mean", "std"]
    )
    assert isinstance(stats, pd.DataFrame)
    return stats


def plot_stats(
    stats: pd.DataFrame,
    title: str = "Comparación de Estadísticas por Configuración",
    img_path: str | None = None,
) -> None:
    stats_reset = stats.reset_index()

    plt.figure(figsize=(15, 8))

    stats_reset["config"] = stats_reset.apply(
        lambda row: f"Mut={row['bitflip probability']}\n"
        f"Cross={row['cross probability']}\n",
        axis=1,
    )

    x_pos = range(len(stats_reset))

    plt.bar(
        x_pos,
        stats_reset["mean"],
        yerr=stats_reset["std"],
        capsize=5,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    plt.xlabel("Configuración", fontsize=11)
    plt.ylabel("Fitness Promedio", fontsize=11)
    plt.xticks(x_pos, stats_reset["config"].tolist(), rotation=45)
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    plt.title(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if img_path:
        print("Saving plot to", img_path)
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
        stats_filename = size_eval_folder / "stats"

        print("Starting evaluations for problem size", problem_size)
        df = eval_results(problem_size, function_evaluations)
        df.to_csv(f"{results_filename}.csv")
        plot_results(
            df,
            title=f"Fitness evolution for different configurations (Size={problem_size})",
            img_path=f"{results_filename}.png",
        )

        print("Generating statistics for problem size", problem_size)
        stats = generate_stats(df)
        stats.to_csv(f"{stats_filename}.csv")
        plot_stats(
            stats,
            title=f"Statistics (Size={problem_size})",
            img_path=f"{stats_filename}.png",
        )


if __name__ == "__main__":
    main()

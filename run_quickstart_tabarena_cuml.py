from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pandas as pd

from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from bencheval.website_format import format_leaderboard


@click.command()
@click.option(
    "--datasets", "-d",
    default="anneal",
    help="Comma-separated list of datasets to run benchmarks on.",
    show_default=True,
)
@click.option(
    "--experiment-name", "-n",
    default="test_rf_model_gpu_anneal",
    help="Name for the experiment.",
    show_default=True,
)
@click.option(
    "--time-limit", "-t",
    default=300,
    type=int,
    help="Time limit in seconds for each experiment (3600 was used in the TabArena 2025 paper).",
    show_default=True,
)
@click.option(
    "--num-bag-folds",
    default=8,
    type=int,
    help="Number of bag folds (8 was used in the TabArena 2025 paper).",
    show_default=True,
)
@click.option(
    "--ignore-cache/--use-cache",
    default=True,
    help="Whether to ignore existing caches and re-run experiments from scratch.",
    show_default=True,
)
def main(
    datasets: str,
    experiment_name: str,
    time_limit: int,
    num_bag_folds: int,
    ignore_cache: bool,
) -> None:
    """Run cuML-accelerated TabArena benchmarks with Random Forest model."""
    expname = str(Path(__file__).parent / "experiments" / "quickstart")  # folder location to save all experiment artifacts
    eval_dir = Path(__file__).parent / "eval" / "quickstart"

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    # Parse comma-separated datasets
    dataset_list = [d.strip() for d in datasets.split(",")]
    
    folds = [0]

    # import your model classes
    from autogluon.tabular.models import RFModel

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
        AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
            # The name you want the config to have
            name=experiment_name,

            # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
            # Supports any model that inherits from `autogluon.core.models.AbstractModel`
            model_cls=RFModel,
            model_hyperparameters={"random_state": None, "ag.ens.use_child_oof": False,
            "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local",},  # uncomment to fit folds sequentially, allowing for use of a debugger
            #"ag_args_fit":{'num_gpus': 0},  # uncomment to run on cpu
            },
            # The non-default model hyperparameters.
            num_bag_folds=num_bag_folds,
            time_limit=time_limit,
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=dataset_list,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    # compute results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    # compare_on_tabarena: Benchmarks your model against ~50 baseline models (LightGBM, XGBoost, LogisticRegression, etc.) by fitting ALL of them on your datasets

    # leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
    #     output_dir=eval_dir,
    #     only_valid_tasks=True,  # True: only compare on tasks ran in `results_lst`
    #     use_model_results=True,  # If False: Will instead use the ensemble/HPO results
    #     new_result_prefix="Demo_",
    # )
    # leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    # print(leaderboard_website.to_markdown(index=False))

if __name__ == "__main__":
    main()

import pickle
import logging
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.scope import render_scope
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)

from empanada.simulator.simulator import (
    SimulatorOutput,
    run_simulator,
)
from empanada.experiments.experiment_definition import (
    ExperimentDefinition,
    SubexperimentDefinition,
)


console = Console()

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)  # NOTE: Change from INFO to DEBUG to see all info

logger = logging.getLogger(__name__)


def get_experiment_file_name(experiment_definition: ExperimentDefinition):
    return Path(f"{hash(experiment_definition)}.experiment_output")


def save_experiment_output(
    experiment_definition,
    experiment_output: list[
        tuple[
            dict[str, Any],
            list[SimulatorOutput],
        ],
    ],
):
    with open(get_experiment_file_name(experiment_definition), "wb") as file_handle:
        pickle.dump(experiment_output, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_saved_experiment_output(experiment_definition: ExperimentDefinition) -> (
    list[
        tuple[
            dict[str, Any],
            list[SimulatorOutput],
        ],
    ]
    | None
):
    file_path = get_experiment_file_name(experiment_definition)
    if not file_path.exists():
        return None
    with open(file_path, "rb") as file_handle:
        return pickle.load(file_handle)


def run_subexperiment(
    subexperiment_definition: SubexperimentDefinition,
    num_replications: int,
    progress: Optional[Progress] = None,
    subexperiment_idx: Optional[int] = None,
):
    replication_sequence = range(num_replications)
    replication_sequence_with_progresss = (
        replication_sequence
        if progress is None
        else progress.track(
            replication_sequence,
            description=(
                "Running replications"
                if subexperiment_idx is None
                else f"Running replications for subexperiment {subexperiment_idx}"
            ),
        )
    )

    return [
        run_simulator(subexperiment_definition.simulator_parameters)
        for _ in replication_sequence_with_progresss
    ]


def run_simulator_for_experiment(
    experiment_definition: ExperimentDefinition,
    progress: Optional[Progress] = None,
    experiment_idx: Optional[int] = None,
) -> list[
    tuple[
        dict[str, Any],
        list[SimulatorOutput],
    ],
]:
    subexperiments = experiment_definition.subexperiments
    subexperiment_list_with_progress = (
        subexperiments
        if progress is None
        else progress.track(
            subexperiments,
            description=(
                "Running subexperiments"
                if experiment_idx is None
                else f"Running subexperiments for experiment {experiment_idx}"
            ),
        )
    )
    return [
        (
            subexperiment_definition.parameters,
            run_subexperiment(
                subexperiment_definition,
                experiment_definition.num_replications,
                progress,
                i,
            ),
        )
        for i, subexperiment_definition in enumerate(subexperiment_list_with_progress)
    ]


def run_experiment(
    experiment_definition: ExperimentDefinition,
    experiment_idx: Optional[int] = None,
    progress: Optional[Progress] = None,
):
    logger.info(
        f"[b]Running experiment{f' {experiment_idx}' if experiment_idx else ''} (hash={hash(experiment_definition)})[/b]",
        extra={"markup": True},
    )
    experiment_output = get_saved_experiment_output(experiment_definition)
    if not experiment_output or experiment_definition.override_rerun_hash_check:
        experiment_output = run_simulator_for_experiment(
            experiment_definition,
            progress,
            experiment_idx,
        )
        save_experiment_output(experiment_definition, experiment_output)

    if experiment_definition.subexperiment_data_analyzer is not None:
        for subexperiment_output in experiment_output:
            experiment_definition.subexperiment_data_analyzer(subexperiment_output)

    if experiment_definition.joint_data_analyzer is not None:
        experiment_definition.joint_data_analyzer(experiment_output)

    return experiment_output
    # ==================== Printing Results ====================
    # console.log(simulator_output.generated_server_args)
    #
    # console.log(simulator_output.results[0])
    #
    # console.log(
    #     render_scope(
    #         {
    #             "Model Name": experiment_definition.simulator_parameters.model_name,
    #             # "Number of Workloads": NUM_WORKLOADS,
    #             "Number of Requests": len(simulator_output.requests),
    #             "Requests per Second": experiment_definition.simulator_parameters.requests_per_second,
    #             "Experiment Time": experiment_definition.simulator_parameters.experiment_time_seconds,
    #         },
    #         title="Experiment Parameters",
    #     )
    # )
    #
    # console.log(simulator_output.benchmark_metrics)
    #
    # run_data_analysis_suite(simulator_output.results)


def run_experiments(experiments: list[ExperimentDefinition]):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        return [
            run_experiment(experiment, i, progress)
            for i, experiment in enumerate(
                progress.track(experiments, description="Running experiments")
            )
        ]

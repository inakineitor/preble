from typing import Any
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.scope import render_scope

from preble.data_parallel_request_cache import (
    DataParallelRequestRouter,
    DataParallelRuntimeSelectionPolicy,
)
from preble.benchmarks.benchmark_workload_gen import (
    DataLoader,
    WorkloadPrefixDataLoader,
)
from preble.benchmarks.exp_configs.model_equations import (
    mistral_7b_A6000_sglang_extend_flashinfer,
    mistrial_7b_A6000_sglang_decode_flashinfer,
)

from empanada.scheduler.empanada_scheduler import EmpanadaScheduler
from empanada.data_loaders.high_variance_workload_prefix_data_loader import (
    HighVarianceWorkloadPrefixDataLoader,
)
from empanada.data_analysis.data_analysis_suite import (
    run_data_analysis_suite,
)
from empanada.simulator.simulator import (
    AcceleratorParameters,
    SimulatorParameters,
    run_simulator,
)
from empanada.experiments.experiment import ExperimentDefinition


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


def run_experiment(experiment: ExperimentDefinition):
    simulator_output = run_simulator(experiment.simulator_parameters)

    # ==================== Printing Results ====================
    console.log(simulator_output.generated_server_args)

    console.log(simulator_output.results[0])

    console.log(
        render_scope(
            {
                "Model Name": experiment.simulator_parameters.model_name,
                # "Number of Workloads": NUM_WORKLOADS,
                "Number of Requests": len(simulator_output.requests),
                "Requests per Second": experiment.simulator_parameters.requests_per_second,
                "Experiment Time": experiment.simulator_parameters.experiment_time_seconds,
            },
            title="Experiment Parameters",
        )
    )

    console.log(simulator_output.benchmark_metrics)

    run_data_analysis_suite(simulator_output.results)


def run_experiments(experiments: list[ExperimentDefinition]):
    return [run_experiment(experiment) for experiment in experiments]

from typing import Any, Callable

import matplotlib


matplotlib.use("module://matplotlib-backend-kitty")

from preble.data_parallel_request_cache import (
    DataParallelRuntimeSelectionPolicy,
    DataParallelRequestRouter,
)
from preble.global_scheduler_with_time import GlobalSchedulerWithTime
from preble.benchmarks.benchmark_workload_gen import (
    DataLoader,
)
from preble.benchmarks.exp_configs.model_equations import (
    mistral_7b_A6000_sglang_extend_flashinfer,
    mistrial_7b_A6000_sglang_decode_flashinfer,
)
from preble.benchmarks.benchmark_workload_gen import WorkloadPrefixDataLoader

from empanada.experiments.experiment_definition import (
    SubexperimentDefinition,
    ExperimentDefinition,
)
from empanada.simulator.simulator import SimulatorParameters, AcceleratorParameters
from empanada.data_loaders.high_variance_workload_prefix_data_loader import (
    HighVarianceWorkloadPrefixDataLoader,
)
from empanada.scheduler.empanada_scheduler import EmpanadaScheduler
from empanada.scripts.experiments.utils.plotting import (
    num_gpus_facet_avg_norm_latency_vs_rps,
    num_gpus_facet_overhead_vs_rps,
    rps_num_gpus_facet_avg_norm_latency_vs_num_workloads,
    rps_num_gpus_facet_avg_norm_latency_vs_output_length_variance,
    rps_facet_avg_norm_latency_vs_num_gpus,
)

RANDOM_SCHEDULER = (
    "Random",
    lambda num_gpus: (
        DataParallelRequestRouter(
            DataParallelRuntimeSelectionPolicy.RANDOM,
            total_nodes=num_gpus,
        )
    ),
)

ROUND_ROBIN_SCHEDULER = (
    "Round Robin",
    lambda num_gpus: (
        DataParallelRequestRouter(
            DataParallelRuntimeSelectionPolicy.ROUND_ROBIN,
            total_nodes=num_gpus,
        )
    ),
)

LEAST_OUTSTANDING_REQUESTS_SCHEDULER = (
    "Least Outstanding Requests",
    lambda num_gpus: (
        DataParallelRequestRouter(
            DataParallelRuntimeSelectionPolicy.LEAST_OUTSTANDING_REQUESTS,
            total_nodes=num_gpus,
        )
    ),
)

PREBLE_SCHEDULER = (
    "Preble Scheduler",
    lambda num_gpus: (
        DataParallelRequestRouter(
            DataParallelRuntimeSelectionPolicy.CUSTOM,
            total_nodes=num_gpus,
            custom_runtime_selector=GlobalSchedulerWithTime(
                num_nodes=num_gpus,
                enable_eviction=False,
                enable_rebalancing=True,
                enable_miss_rate=True,
            ),
        )
    ),
)

EMPANADA_SCHEDULER = (
    "Empanada Scheduler",
    lambda num_gpus: (
        DataParallelRequestRouter(
            DataParallelRuntimeSelectionPolicy.CUSTOM,
            total_nodes=num_gpus,
            custom_runtime_selector=EmpanadaScheduler(
                num_nodes=num_gpus,
                enable_eviction=False,
                enable_rebalancing=True,
                enable_miss_rate=True,
            ),
        )
    ),
)

# ==================== Caches ====================

# CACHE: Works for finding the best performing naive method to later compare. This is for findind X
# NOTE: Run.
# NUM_WORKLOADS_OPTIONS = [10]
# NUM_IN_CONTEXT_EXAMPLES_OPTIONS = [4]
# OUTPUT_LENGTH_DISTRIBUTION_OPTIONS = [[(1.0, 150)]]
# NUM_GPUS_OPTIONS = [2**2, 2**3, 2**4]
# REQUESTS_PER_SECOND_OPTIONS = [2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7]
# CREATE_ROUTER_OPTIONS = [
#     RANDOM_SCHEDULER,
#     ROUND_ROBIN_SCHEDULER,
#     LEAST_OUTSTANDING_REQUESTS_SCHEDULER,
#     PREBLE_SCHEDULER,
# ]
# DATA_ANALYZER = num_gpus_facet_avg_norm_latency_vs_rps

# CACHE: For showing that Preble does not run better than X for high RPS across workloads
# FIX: Run. Plot must be changed.
# NUM_WORKLOADS_OPTIONS = [1, 5, 10]
# NUM_IN_CONTEXT_EXAMPLES_OPTIONS = [4]
# OUTPUT_LENGTH_DISTRIBUTION_OPTIONS = [[(1.0, 150)]]
# NUM_GPUS_OPTIONS = [2**2, 2**3, 2**4]
# REQUESTS_PER_SECOND_OPTIONS = [2**5, 2**6, 2**7]
# CREATE_ROUTER_OPTIONS = [LEAST_OUTSTANDING_REQUESTS_SCHEDULER, PREBLE_SCHEDULER]
# DATA_ANALYZER = rps_num_gpus_facet_avg_norm_latency_vs_num_workloads

# CACHE: For showing that Preble average normalized latency gets worse as as the number of GPUs increase. Test with generally high RPS.
# NOTE: Run.
# PLOT: Each facet is the number of GPUs and within each facet the the x-axis is the RPS and the y-axis is the average normalized latency.
# NUM_WORKLOADS_OPTIONS = [10]
# NUM_IN_CONTEXT_EXAMPLES_OPTIONS = [4]
# OUTPUT_LENGTH_DISTRIBUTION_OPTIONS = [[(1.0, 150)]]
# NUM_GPUS_OPTIONS = [2**2, 2**3, 2**4]
# REQUESTS_PER_SECOND_OPTIONS = [2**3, 2**4, 2**5, 2**6, 2**7]
# CREATE_ROUTER_OPTIONS = [LEAST_OUTSTANDING_REQUESTS_SCHEDULER, PREBLE_SCHEDULER]
# DATA_ANALYZER = num_gpus_facet_avg_norm_latency_vs_rps

# CACHE: For showing that Preble overhead gets worse as as the number of GPUs increase. Test with generally high RPS.
# NOTE: Run.
# PLOT: Each facet is the number of GPUs and within each facet the the x-axis is the RPS and the y-axis is the average scheduling overhead.
# NUM_WORKLOADS_OPTIONS = [10]
# NUM_IN_CONTEXT_EXAMPLES_OPTIONS = [4]
# OUTPUT_LENGTH_DISTRIBUTION_OPTIONS = [[(1.0, 150)]]
# NUM_GPUS_OPTIONS = [2**3, 2**4, 2**5]
# REQUESTS_PER_SECOND_OPTIONS = [2**3, 2**4, 2**5, 2**6, 2**7]
# CREATE_ROUTER_OPTIONS = [LEAST_OUTSTANDING_REQUESTS_SCHEDULER, PREBLE_SCHEDULER]
# DATA_ANALYZER = num_gpus_facet_overhead_vs_rps

# CACHE: For showing that length distributions adversely affect Preble
# HACK: Run. Suboptimal plot.
# PLOT: The facet is the number of GPUs. The x-axis is the the variance of the distribution and the y-axis is the average normalized latency.
# NUM_WORKLOADS_OPTIONS = [10]
# NUM_IN_CONTEXT_EXAMPLES_OPTIONS = [4]
# OUTPUT_LENGTH_DISTRIBUTION_OPTIONS = [
#     [(1.0, 150)],
#     [(1 / 3, 1), (1 / 3, 150), (1 / 3, 300)],
#     [(0.5, 1), (0.5, 300)],
# ]
# NUM_GPUS_OPTIONS = [2**3, 2**4]
# REQUESTS_PER_SECOND_OPTIONS = [2**3, 2**4, 2**5, 2**6]
# CREATE_ROUTER_OPTIONS = [LEAST_OUTSTANDING_REQUESTS_SCHEDULER, PREBLE_SCHEDULER]
# DATA_ANALYZER = rps_num_gpus_facet_avg_norm_latency_vs_output_length_variance


# CACHE: For showing that length distributions adversely affect Preble. All schedulers.
# NOTE: Run. Best plot. Saved.
# PLOT: The facet is the number of GPUs. The x-axis is the the variance of the distribution and the y-axis is the average normalized latency.
NUM_WORKLOADS_OPTIONS = [10]
NUM_IN_CONTEXT_EXAMPLES_OPTIONS = [4]
OUTPUT_LENGTH_DISTRIBUTION_OPTIONS = [
    [(1.0, 150)],
    [(1 / 3, 1), (1 / 3, 150), (1 / 3, 300)],
    [(0.5, 1), (0.5, 300)],
]
NUM_GPUS_OPTIONS = [2**3, 2**4]
REQUESTS_PER_SECOND_OPTIONS = [2**5, 2**6]
CREATE_ROUTER_OPTIONS = [
    RANDOM_SCHEDULER,
    ROUND_ROBIN_SCHEDULER,
    LEAST_OUTSTANDING_REQUESTS_SCHEDULER,
    PREBLE_SCHEDULER,
]
DATA_ANALYZER = rps_num_gpus_facet_avg_norm_latency_vs_output_length_variance

# ==================== Beginning of Simulation Code ====================

# ==================== Dataloader Parameters ====================
# NUM_WORKLOADS_OPTIONS = [10]
# NUM_IN_CONTEXT_EXAMPLES_OPTIONS = [4]
# OUTPUT_LENGTH_DISTRIBUTION_OPTIONS = [
#     [(1.0, 150)],
#     [
#         (0.5, 1),
#         (0.5, 300),
#     ],
# ]

# ==================== Accelerator Parameters ====================
# NUM_GPUS_OPTIONS = [
#     2**1,
#     2**2,
#     2**3,
#     2**4,
# ]
KV_CACHE_MEMORY = 131072 * 198516  # A6000 simulator configuration used in experiments
FORWARD_SIMULATION_EXTEND = mistral_7b_A6000_sglang_extend_flashinfer
FORWARD_SIMULATION_DECODE = mistrial_7b_A6000_sglang_decode_flashinfer

# ==================== Simulator Parameters ====================
# REQUESTS_PER_SECOND_OPTIONS = [
#     2**-1,
#     2**0,
#     2**1,
#     2**2,
#     2**3,
#     2**4,
#     2**5,
# ]
EXPERIMENT_TIME_SECONDS = 30

# ==================== Server Parameters ====================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"


# CREATE_ROUTER_OPTIONS: list[tuple[str, Callable[[int], DataParallelRequestRouter]]] = [
#     RANDOM_SCHEDULER,
#     ROUND_ROBIN_SCHEDULER,
#     LEAST_OUTSTANDING_REQUESTS_SCHEDULER,
#     PREBLE_SCHEDULER,
#     EMPANADA_SCHEDULER,
# ]


def create_variable_length_data_loader(
    num_workloads: int,
    num_requests: int,
    num_in_context_examples: int,
    output_length_distribution: list[tuple[float, int]],
    tokenizer: Any,
) -> DataLoader:
    return HighVarianceWorkloadPrefixDataLoader(
        num_workloads,
        num_requests,
        tokenizer,
        num_in_context_examples=num_in_context_examples,
        output_length_distribution=output_length_distribution,
    )


def create_subexperiment(
    num_gpus: int,
    router_name: str,
    create_router: Callable[[int], DataParallelRequestRouter],
    requests_per_second: int,
    num_workloads: int,
    num_in_context_examples: int,
    output_length_distribution: list[tuple[float, int]],
):
    simulator_parameters = SimulatorParameters(
        accelerator_parameters=AcceleratorParameters(
            num_gpus=num_gpus,
            kv_cache_memory=KV_CACHE_MEMORY,
            forward_simulation_extend=FORWARD_SIMULATION_EXTEND,
            forward_simulation_decode=FORWARD_SIMULATION_DECODE,
        ),
        create_data_parallel_request_router=create_router,
        create_data_loader=create_variable_length_data_loader,
        requests_per_second=requests_per_second,
        num_workloads=num_workloads,
        num_in_context_examples=num_in_context_examples,
        output_length_distribution=output_length_distribution,
        experiment_time_seconds=EXPERIMENT_TIME_SECONDS,
        model_name=MODEL_NAME,
    )
    return SubexperimentDefinition(
        parameters={
            "num_gpus": num_gpus,
            "router_name": router_name,
            "requests_per_second": requests_per_second,
            "num_workloads": num_workloads,
            "num_in_context_examples": num_in_context_examples,
            "output_length_distribution": output_length_distribution,
        },
        simulator_parameters=simulator_parameters,
    )


experiments = [
    ExperimentDefinition(
        simulate=True,
        num_replications=1,
        subexperiments=[
            create_subexperiment(
                num_gpus,
                router_name,
                create_router,
                requests_per_second,
                num_workloads,
                num_in_context_examples,
                output_length_distribution,
            )
            for num_gpus in NUM_GPUS_OPTIONS
            for router_name, create_router in CREATE_ROUTER_OPTIONS
            for requests_per_second in REQUESTS_PER_SECOND_OPTIONS
            for num_workloads in NUM_WORKLOADS_OPTIONS
            for num_in_context_examples in NUM_IN_CONTEXT_EXAMPLES_OPTIONS
            for output_length_distribution in OUTPUT_LENGTH_DISTRIBUTION_OPTIONS
        ],
        joint_data_analyzer=DATA_ANALYZER,
        # override_rerun_hash_check=True,
    ),
]

# def create_fixed_length_data_loader(
#     num_workloads: int,
#     num_requests: int,
#     num_in_context_examples: int,
#     output_length_distribution: list[tuple[float, int]],
#     tokenizer: Any,
# ) -> DataLoader:
#     return WorkloadPrefixDataLoader(
#         num_workloads,
#         num_requests,
#         tokenizer,
#         num_in_context_examples=num_in_context_examples,
#         output_len=output_length_distribution[0][1],
#     )

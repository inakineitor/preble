from typing import Any

from preble.data_parallel_request_cache import (
    DataParallelRuntimeSelectionPolicy,
    DataParallelRequestRouter,
)
from preble.benchmarks.benchmark_workload_gen import (
    DataLoader,
)
from preble.benchmarks.exp_configs.model_equations import (
    mistral_7b_A6000_sglang_extend_flashinfer,
    mistrial_7b_A6000_sglang_decode_flashinfer,
)

from empanada.experiments.experiment import ExperimentDefinition
from empanada.simulator.simulator import SimulatorParameters, AcceleratorParameters
from empanada.data_loaders.high_variance_workload_prefix_data_loader import (
    HighVarianceWorkloadPrefixDataLoader,
)
from empanada.scheduler.empanada_scheduler import EmpanadaScheduler

# ==================== Dataloader Parameters ====================
NUM_WORKLOADS = 10
NUM_IN_CONTEXT_EXAMPLES = 4
OUTPUT_LENGTH = 100  # For fixed length dataloaders
OUTPUT_LENGTH_DISTRIBUTION = [
    (0.5, 1),
    (0.5, 200),
]  # For variable length distributions

# ==================== Accelerator Parameters ====================
NUM_GPUS = 8
KV_CACHE_MEMORY = 131072 * 198516  # A6000 simulator configuration used in experiments
FORWARD_SIMULATION_EXTEND = mistral_7b_A6000_sglang_extend_flashinfer
FORWARD_SIMULATION_DECODE = mistrial_7b_A6000_sglang_decode_flashinfer

# ==================== Simulator Parameters ====================
REQUESTS_PER_SECOND = 1
EXPERIMENT_TIME_SECONDS = 30

# ==================== Server Parameters ====================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"


def create_data_loader(num_requests: int, tokenizer: Any) -> DataLoader:
    # return WorkloadPrefixDataLoader(
    #     NUM_WORKLOADS,
    #     num_requests,
    #     tokenizer,
    #     num_in_context_examples=NUM_IN_CONTEXT_EXAMPLES,
    #     output_len=OUTPUT_LENGTH,
    # )
    return HighVarianceWorkloadPrefixDataLoader(
        NUM_WORKLOADS,
        num_requests,
        tokenizer,
        num_in_context_examples=NUM_IN_CONTEXT_EXAMPLES,
        output_length_distribution=OUTPUT_LENGTH_DISTRIBUTION,
    )


def create_router(num_gpus: int):
    return DataParallelRequestRouter(
        DataParallelRuntimeSelectionPolicy.CUSTOM,
        total_nodes=num_gpus,
        custom_runtime_selector=EmpanadaScheduler(
            num_nodes=num_gpus,
            enable_eviction=False,
            enable_rebalancing=True,
            enable_miss_rate=True,
        ),
    )


simulator_parameters = SimulatorParameters(
    accelerator_parameters=AcceleratorParameters(
        num_gpus=NUM_GPUS,
        kv_cache_memory=KV_CACHE_MEMORY,
        forward_simulation_extend=FORWARD_SIMULATION_EXTEND,
        forward_simulation_decode=FORWARD_SIMULATION_DECODE,
    ),
    create_data_parallel_request_router=create_router,
    create_data_loader=create_data_loader,
    requests_per_second=REQUESTS_PER_SECOND,
    experiment_time_seconds=EXPERIMENT_TIME_SECONDS,
    model_name=MODEL_NAME,
)

experiments = [
    ExperimentDefinition(
        experiment_name="test experiment",
        simulate=True,
        simulator_parameters=simulator_parameters,
    )
]

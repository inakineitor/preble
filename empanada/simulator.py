from typing import List, Optional, Union
import uuid
import logging
import random

from rich.console import Console
from rich.logging import RichHandler
from rich.scope import render_scope
from transformers import AutoTokenizer
import numpy as np

from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.router.model_runner import GPUConfig


from preble.data_parallel_request_cache import (
    DataParallelRequestRouter,
    DataParallelRuntimeSelectionPolicy,
)
from preble.benchmarks.benchmark_workload_gen import WorkloadPrefixDataLoader
from preble.benchmarks.benchmark_utils import BenchmarkMetrics
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
from empanada.simulator.server_runtime_simulator import (
    ServerRuntimeSimulator,
)
from empanada.simulator.simulation import Simulation


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


def random_uuid_string():
    return str(uuid.uuid4().hex)


def create_simulator_args(
    model_path: str,
    profile_mode: bool = False,
    tokenizer_path: Optional[str] = None,
    load_format: str = "auto",
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = True,
    mem_fraction_static: float = ServerArgs.mem_fraction_static,
    max_prefill_num_token: int = ServerArgs.max_prefill_num_token,
    context_length: int = ServerArgs.context_length,
    tp_size: int = 1,
    schedule_heuristic: str = "lpm",
    attention_reduce_in_fp32: bool = False,
    random_seed: int = 42,
    log_level: str = "error",
    disable_radix_cache: bool = False,
    enable_flashinfer: bool = False,
    disable_regex_jump_forward: bool = False,
    disable_disk_cache: bool = False,
    api_key: str = "",
    port: Optional[int] = None,
    additional_ports: Optional[Union[List[int], int]] = None,
    cuda_devices: Optional[List[int]] = None,
    freeze: bool = False,
    log_prefix_hit: bool = False,
    chunk_prefill_budget: int = 0,
    hit_trace_window_size: int = 30,  # seconds
    report_hit_ratio: bool = True,
    enable_iterative_eviction: bool = False,
    enable_partial_eviction: bool = False,
) -> tuple[bool, ServerArgs]:
    host = "0.0.0.0"
    port, additional_ports = 0, [0] * 100
    if profile_mode:
        load_format = "dummy"
    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        host=host,
        port=port,
        additional_ports=additional_ports,
        load_format=load_format,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
        mem_fraction_static=mem_fraction_static,
        max_prefill_num_token=max_prefill_num_token,
        context_length=context_length,
        tp_size=tp_size,
        schedule_heuristic=schedule_heuristic,
        attention_reduce_in_fp32=attention_reduce_in_fp32,
        random_seed=random_seed,
        log_level=log_level,
        cuda_devices=cuda_devices,
        freeze=freeze,
        log_prefix_hit=log_prefix_hit,
        disable_radix_cache=disable_radix_cache,
        enable_flashinfer=enable_flashinfer,
        disable_regex_jump_forward=disable_regex_jump_forward,
        disable_disk_cache=disable_disk_cache,
        api_key=api_key,
        chunk_prefill_budget=chunk_prefill_budget,
        hit_trace_window_size=hit_trace_window_size,
        report_hit_ratio=report_hit_ratio,
        enable_iterative_eviction=enable_iterative_eviction,
        enable_partial_eviction=enable_partial_eviction,
    )
    return profile_mode, server_args


def create_gpu(
    id: int,
    server_args: ServerArgs,
    forward_simulation_extend,
    forward_simulation_decode,
    kv_cache_memory,
):
    gpu_config = GPUConfig(gpu_id=id, url=None, use_ssh=False, runtime_args=server_args)
    gpu_config.regist_simulator_config(
        forward_simulation=[forward_simulation_extend, forward_simulation_decode],
        kv_cache_memory=kv_cache_memory,
        lp_forward_simulation=None,
    )
    return gpu_config


if __name__ == "__main__":

    # Set random seeds
    random.seed(2333)
    np.random.seed(2333)

    # ==================== Simulator Parameters ====================
    REQUESTS_PER_SECOND = 1
    EXPERIMENT_TIME_SECONDS = 30

    # ==================== Dataloader Parameters ====================
    NUM_WORKLOADS = 10
    NUM_IN_CONTEXT_EXAMPLES = 4
    OUTPUT_LENGTH = 100  # For fixed length dataloaders
    OUTPUT_LENGTH_DISTRIBUTION = [
        (0.5, 1),
        (0.5, 200),
    ]  # For varaible length distributions

    # ==================== Accelerator Parameters ====================
    NUM_GPUS = 8
    KV_CACHE_MEMORY = (
        131072 * 198516
    )  # A6000 simulator configuration used in experiments
    FORWARD_SIMULATION_EXTEND = mistral_7b_A6000_sglang_extend_flashinfer
    FORWARD_SIMULATION_DECODE = mistrial_7b_A6000_sglang_decode_flashinfer

    # ==================== Server Parameters ====================
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"

    # ==================== Computed Simulator Parameters ====================
    num_requests = int(REQUESTS_PER_SECOND * EXPERIMENT_TIME_SECONDS)
    profile_mode, server_args = create_simulator_args(model_path=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ==================== Computed Dataloader Parameters ====================
    # NOTE: Original data loader
    # dataloader = WorkloadPrefixDataLoader(
    #     NUM_WORKLOADS,
    #     num_requests,
    #     tokenizer,
    #     num_in_context_examples=NUM_IN_CONTEXT_EXAMPLES,
    #     output_len=OUTPUT_LENGTH,
    # )
    dataloader = HighVarianceWorkloadPrefixDataLoader(
        NUM_WORKLOADS,
        num_requests,
        tokenizer,
        num_in_context_examples=NUM_IN_CONTEXT_EXAMPLES,
        output_length_distribution=OUTPUT_LENGTH_DISTRIBUTION,
    )
    requests = dataloader.generate_workload(k=1)  # `k` is unused parameter

    # ==================== Computed Accelerator Parameters ====================
    gpu_configs = [
        create_gpu(
            gpu_id,
            server_args,
            FORWARD_SIMULATION_EXTEND,
            FORWARD_SIMULATION_DECODE,
            KV_CACHE_MEMORY,
        )
        for gpu_id in range(NUM_GPUS)
    ]

    # ==================== Computed Server Parameters ====================
    runtimes = [
        ServerRuntimeSimulator(
            gpu_config=config,
            server_args=server_args,
            profile_mode=profile_mode,
        )
        for config in gpu_configs
    ]
    # vocab_size = runtimes[0].model_rpc.model_config.vocab_size
    router = DataParallelRequestRouter(
        DataParallelRuntimeSelectionPolicy.CUSTOM,
        total_nodes=NUM_GPUS,
        custom_runtime_selector=EmpanadaScheduler(
            num_nodes=NUM_GPUS,
            enable_eviction=False,
            enable_rebalancing=True,
            enable_miss_rate=True,
        ),
    )

    console.log(server_args)

    # ==================== Setting Up Simulator ====================
    simulator = Simulation(runtimes, router)
    simulator.initialize_all_request_with_rps(
        requests, REQUESTS_PER_SECOND, EXPERIMENT_TIME_SECONDS
    )
    simulator.start_model_forwarding_loop()

    results = simulator.run()

    console.log(results[0])
    # console.log(results[1])
    # console.log(results[2])
    # console.log(results[3])

    # ==================== Processing Benchmarks ====================
    bench_metrics = BenchmarkMetrics.gen_benchmark_metrics(
        tokenizer=tokenizer,
        req_func_outputs=results,
        overall_latency=EXPERIMENT_TIME_SECONDS,
        time_limit=EXPERIMENT_TIME_SECONDS,
        gpu_counts=NUM_GPUS,
    )

    # ==================== Printing Results ====================
    console.log(
        render_scope(
            {
                "Model Name": MODEL_NAME,
                "Number of Workloads": NUM_WORKLOADS,
                "IDK what this is": 0,
                "Number of Requests": num_requests,
                "Requests per Second": REQUESTS_PER_SECOND,
                "Experiment Time": EXPERIMENT_TIME_SECONDS,
            },
            title="Experiment Parameters",
        )
    )
    console.log(bench_metrics)

    run_data_analysis_suite(results)

    # bench_metrics.to_log_file(exp_params)

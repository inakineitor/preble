from typing import Any, Callable, List, Optional, Union
import uuid
import logging
import random
from dataclasses import dataclass, field
import hashlib

from transformers import AutoTokenizer
import numpy as np
import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.router.model_runner import GPUConfig


from preble.data_parallel_request_cache import (
    DataParallelRequestRouter,
)
from preble.benchmarks.benchmark_workload_gen import (
    DataLoader,
)
from preble.benchmarks.benchmark_utils import BenchmarkMetrics, RequestFuncOutput

from empanada.simulator.server_runtime_simulator import (
    ServerRuntimeSimulator,
)
from empanada.simulator.simulation import Simulation


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


@dataclass(eq=True, frozen=True)
class AcceleratorParameters:
    num_gpus: int
    kv_cache_memory: int  # number of bytes
    forward_simulation_extend: Callable[
        [int, int, int, list[int], Optional[int], Optional[torch.Tensor]], float
    ] = field(compare=False, hash=False)
    forward_simulation_decode: Callable[[int, int, int, Optional[int]], float] = field(
        compare=False, hash=False
    )


@dataclass(eq=True, frozen=True)
class SimulatorParameters:
    accelerator_parameters: AcceleratorParameters
    create_data_parallel_request_router: Callable[[int], DataParallelRequestRouter] = (
        field(hash=False)
    )  # (num_gpus: int) -> DataParallelRequestRouter
    create_data_loader: Callable[
        [int, int, int, list[tuple[float, int]], list[tuple[float, int]], Any],
        DataLoader,
    ] = field(hash=False)
    # (
    #   num_workloads: int,
    #   num_requests: int,
    #   num_in_context_examples: int,
    #   output_length_distribution: list[tuple[float, int]],
    #   tokenizer: Any,
    # ) -> DataLoader
    requests_per_second: float
    num_workloads: int
    num_in_context_examples: int
    output_length_distribution: list[tuple[float, int]] = field(hash=False)
    max_new_tokens_distribution: list[tuple[float, int]] = field(hash=False)
    experiment_time_seconds: int
    model_name: str

    def __hash__(self) -> int:
        return hash(
            (
                self.accelerator_parameters,
                self.requests_per_second,
                self.num_workloads,
                self.num_in_context_examples,
                self.experiment_time_seconds,
                tuple(self.output_length_distribution),
                tuple(self.max_new_tokens_distribution),
                int.from_bytes(
                    hashlib.sha256(self.model_name.encode("utf-8")).digest(),
                    "big",  # strings in python get salted by default hash so we must use stable cryptographic hash
                ),
            )
        )


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


def create_gpus(server_args: ServerArgs, accelerator_parameters: AcceleratorParameters):
    return [
        create_gpu(
            gpu_id,
            server_args,
            accelerator_parameters.forward_simulation_extend,
            accelerator_parameters.forward_simulation_decode,
            accelerator_parameters.kv_cache_memory,
        )
        for gpu_id in range(accelerator_parameters.num_gpus)
    ]


@dataclass
class SimulatorOutput:
    requests: list[Any]  # TODO: Replace
    results: list[RequestFuncOutput]
    benchmark_metrics: BenchmarkMetrics
    generated_server_args: ServerArgs


def run_simulator(simulator_parameters: SimulatorParameters) -> SimulatorOutput:
    # Set random seeds
    random.seed(2333)
    np.random.seed(2333)

    # ==================== Computed Simulator Parameters ====================
    num_requests = int(
        simulator_parameters.requests_per_second
        * simulator_parameters.experiment_time_seconds
    )
    profile_mode, server_args = create_simulator_args(
        model_path=simulator_parameters.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(simulator_parameters.model_name)

    # ==================== Computed Dataloader Parameters ====================
    dataloader = simulator_parameters.create_data_loader(
        simulator_parameters.num_workloads,
        num_requests,
        simulator_parameters.num_in_context_examples,
        simulator_parameters.output_length_distribution,
        simulator_parameters.max_new_tokens_distribution,
        tokenizer,
    )
    requests = dataloader.generate_workload()

    # ==================== Computed Accelerator Parameters ====================
    gpu_configs = create_gpus(server_args, simulator_parameters.accelerator_parameters)

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
    router = simulator_parameters.create_data_parallel_request_router(
        simulator_parameters.accelerator_parameters.num_gpus
    )

    # ==================== Setting Up Simulator ====================
    simulator = Simulation(runtimes, router)
    simulator.initialize_all_request_with_rps(
        requests,
        simulator_parameters.requests_per_second,
        simulator_parameters.experiment_time_seconds,
    )
    simulator.start_model_forwarding_loop()

    results = simulator.run()

    # ==================== Processing Benchmarks ====================
    bench_metrics = BenchmarkMetrics.gen_benchmark_metrics(
        tokenizer=tokenizer,
        req_func_outputs=results,
        overall_latency=simulator_parameters.experiment_time_seconds,
        time_limit=simulator_parameters.experiment_time_seconds,
        gpu_counts=simulator_parameters.accelerator_parameters.num_gpus,
    )

    # bench_metrics.to_log_file(exp_params)

    return SimulatorOutput(
        requests=requests,
        results=results,
        benchmark_metrics=bench_metrics,
        generated_server_args=server_args,
    )

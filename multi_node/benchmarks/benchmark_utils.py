from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import logging
import uuid
from dataclasses import field
import json
import sys, os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sglang.srt.managers.router.model_runner import GPUConfig
from benchmark_workload_gen import (
    DataLoader,
    ToolBenchDataLoader,
    RandomDataLoader,
    LoadDistribution,
    Oracle,
    TBOracle,
    TBOracleB,
    LooGLEDataset,
    LooGLEDatasetType,
    LoogleOracle,
)
from sglang.srt.server_args import ServerArgs

@dataclass
class WorkloadConfig:
    num_prefix_patterns: int
    random_ratio: float
    num_requests: int
    request_rate: float
    requests: List[Dict]
    dataloader: DataLoader
    exp_time: Optional[float] = float("inf")

    def __repr__(self) -> str:
        return (
            f"=====STARTING BENCHMARK OF {self.num_prefix_patterns} WORKLOADS, "
            f'{self.random_ratio} NON-SHARED, '
            f'{self.num_requests} REQUESTS, '
            f'{self.request_rate} REQ/s, '
            f'{self.exp_time} seconds ====='
        )

@dataclass
class MajorExperimentArgs:
    runtime_args: Dict
    workload_configs: List[WorkloadConfig]
    gpu_configs: List[GPUConfig]
    simulate: bool
    log_file_path: str
    selector_configs: List

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    request_latency: float = 0
    ttft: float = 0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    global_time: float = 0
    output_len: float = None
    tpot: float = None
    prefill_decode_ratio: float = None
    send_out_time: float = 0.0
    route_dest: int = None

    def update_metrics(
        self,
        tokenizer,
    ):
        # In simulation this will be set
        if self.output_len is None:
            self.output_len = len(tokenizer(self.generated_text).input_ids)
        # print(self.output_len, self.generated_text, self.success, self.error)
        if self.output_len > 1:
            self.tpot = (self.request_latency - self.ttft) / (self.output_len - 1)
        self.prefill_decode_ratio = self.ttft / self.request_latency

    @property
    def total_tokens(self):
        return self.prompt_len + self.output_len

    @property
    def overall_throughput(self):
        return self.total_tokens / self.request_latency

    def to_json(self):
        return json.dumps(self.__dict__)


@dataclass
class BenchmarkMetrics:
    num_finished_requests: int
    average_finished_topt: float
    ttfts: List[float]
    tpots: List[float]
    throughput_tok_sec: float
    all_results: List[RequestFuncOutput]
    average_request_latency: float
    std_request_latency: float
    average_p90: float
    max_latency: float
    p99_latency: float
    average_ttft: float
    average_topt: float
    prefill_decode_ratio: float
    overall_latency: float
    requests_per_sec: float
    gpu_counts: Dict[int, int]

    def gen_benchmark_metrics(
        tokenizer,
        req_func_outputs: List[RequestFuncOutput],
        overall_latency: float,
        time_limit: int = 100,
        gpu_counts={},
    ):
        for result in req_func_outputs:
            result.update_metrics(tokenizer)  # Computes the generated output tokens

        ttfts = [result.ttft for result in req_func_outputs]
        tpots = [result.tpot for result in req_func_outputs if result.tpot is not None]
        overall_latency = overall_latency
        request_latencies = [result.request_latency for result in req_func_outputs]
        throughput_tok_sec = (
            sum([result.total_tokens for result in req_func_outputs]) / overall_latency
        )
        all_results = req_func_outputs

        num_finished_requests = sum(
            [
                result.global_time <= time_limit
                for result in req_func_outputs
                if result.success
            ]
        )
        average_finished_tpot = np.average(
            [
                result.tpot
                for result in req_func_outputs
                if result.tpot is not None
                and result.global_time <= time_limit
                and result.success
            ]
        )
        prefill_decode_ratio = np.average(
            [result.prefill_decode_ratio for result in req_func_outputs]
        )

        average_request_latency, std_request_latency, average_p90 = (
            np.mean(request_latencies),
            np.std(request_latencies),
            np.percentile(request_latencies, 90),
        )
        max_latency, p99_latency = np.max(request_latencies), np.percentile(
            request_latencies, 99
        )
        average_ttft = np.mean(ttfts)
        average_topt = np.mean(tpots)
        requests_per_sec = len(req_func_outputs) / overall_latency
        return BenchmarkMetrics(
            num_finished_requests=num_finished_requests,
            average_finished_topt=average_finished_tpot,
            ttfts=ttfts,
            tpots=tpots,
            throughput_tok_sec=throughput_tok_sec,
            all_results=all_results,
            average_request_latency=average_request_latency,
            overall_latency=overall_latency,
            std_request_latency=std_request_latency,
            average_p90=average_p90,
            max_latency=max_latency,
            p99_latency=p99_latency,
            average_ttft=average_ttft,
            average_topt=average_topt,
            prefill_decode_ratio=prefill_decode_ratio,
            requests_per_sec=requests_per_sec,
            gpu_counts=gpu_counts,
        )

    @property
    def num_sucessful_requests(self):
        return sum([1 if result.success else 0 for result in self.all_results])

    def to_json(self):
        all_reqs = [result.to_json() for result in self.all_results]
        return {
            "num_finished_requests": self.num_finished_requests,
            "average_finished_topt": self.average_finished_topt,
            "ttfts": self.ttfts,
            "tpots": self.tpots,
            "overall_latency": self.overall_latency,
            "average_request_latency": self.average_request_latency,
            "std_request_latency": self.std_request_latency,
            "average_p90": self.average_p90,
            "max_latency": self.max_latency,
            "p99_latency": self.p99_latency,
            "average_ttft": self.average_ttft,
            "average_topt": self.average_topt,
            "throughput_tok_sec": self.throughput_tok_sec,
            "all_reqs": all_reqs,
            "prefill_decode_ratio": self.prefill_decode_ratio,
        }

    def to_log_file(self, exp_params):
        logging.debug(f"Params=({exp_params}) Overall Latency: {self.overall_latency}")
        logging.debug(
            f"Params=({exp_params}) Overall Throughput: {self.requests_per_sec}"
        )
        logging.debug(
            f"Params=({exp_params}) Overall Request Latency: {self.average_request_latency}, STD: {self.std_request_latency}, P90: {self.average_p90}"
        )
        logging.debug(
            f"Params=({exp_params}) Average TTFT: {self.average_ttft}, Average TOPT: {self.average_topt}, Throughput ToksPerSec: {self.throughput_tok_sec}"
        )
        logging.debug(
            f"Params=({exp_params}) Num Finished Requests: {self.num_finished_requests}, Finished Throughput ToksPerSec: {self.average_finished_topt}"
        )
        logging.debug(
            f"Params=({exp_params}) Overall Max Latency: {self.max_latency}, P99: {self.p99_latency}"
        )
        logging.debug(
            f"Params=({exp_params}) Overall PrefillRatio: {self.prefill_decode_ratio}"
        )
        logging.debug(f"Params=({exp_params}) Counts: {self.gpu_counts}")

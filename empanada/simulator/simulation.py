"""Modify max time"""

from typing import Dict
import json
import heapq
import numpy as np
import logging
from dataclasses import dataclass, asdict

import numpy as np

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
)

from preble.data_parallel_request_cache import (
    DataParallelRequestRouter,
)
from preble.benchmarks.benchmark_utils import RequestFuncOutput

from empanada.simulator.server_runtime_simulator import (
    ServerRuntimeSimulator,
)
import empanada.simulator.simulation_events.simulation_event as simulation_event
from empanada.simulator.simulation_events.send_request_event import (
    SendRequestEvent,
)
from empanada.simulator.simulation_events.generate_request_event import (
    GenerateRequestEvent,
)
from empanada.simulator.simulation_events.model_step_event import (
    ModelStepEvent,
)


@dataclass
class SimulationParameters:
    runtimes: list[ServerRuntimeSimulator]
    router: DataParallelRequestRouter


class Simulation:
    def __init__(
        self, runtimes: list[ServerRuntimeSimulator], router: DataParallelRequestRouter
    ):
        self.global_clock = 0.0
        self.runtimes: list[ServerRuntimeSimulator] = runtimes
        self.router = router
        self.events = []
        # rid -> RequestFuncOutput
        self.request_output: Dict[str, RequestFuncOutput] = {}
        self.unfinished_requests = 0
        self.rid_to_input = {}  # rid -> input request

    @classmethod
    def from_simulation_parameters(cls, simulation_parameters: SimulationParameters):
        return cls(
            runtimes=simulation_parameters.runtimes, router=simulation_parameters.router
        )

    def add_event(self, event: simulation_event.SimulationEvent):
        heapq.heappush(self.events, event)

    def reset_state(self):
        self.global_clock = 0.0
        self.events = []
        self.request_output = {}
        for runtime in self.runtimes:
            runtime.reset_clock()

    def warm_up(self):
        logging.info("--- Warm up started ---")
        prompt = "Say this is a warmup request."
        warm_up_request = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "true_output_length": 16,
                "max_new_tokens": 16,
            },
            "input_ids": [0] * 9,
        }
        for i in range(len(self.runtimes)):
            self.request_output[str(i)] = RequestFuncOutput(
                prompt_len=len(warm_up_request["input_ids"]),
                send_out_time=0.0,
                route_dest=i,
            )
            generate_input = GenerateReqInput(
                text=prompt,
                sampling_params=warm_up_request["sampling_params"],
                rid=str(i),
                stream=True,
            )
            GenerateRequestEvent(0.0, generate_input, i).process_event(self)
        while True:
            if all(r.success for r in self.request_output.values()):
                break
            for i in range(len(self.runtimes)):
                ModelStepEvent(0.0, i).process_event(self)
        self.reset_state()
        logging.info("--- Warm up finished ---")

    def run(self) -> list[RequestFuncOutput]:
        previous_stamp = self.global_clock
        while self.events:
            if self.global_clock > self.time_litmit:
                break
            if not self.unfinished_requests:
                break
            if self.global_clock - previous_stamp >= 10.0:
                logging.info(
                    f"------------ Global clock: {self.global_clock}, "
                    f"finished: {len([r for r in self.request_output.values() if r.success])} "
                    f" / {len(self.request_output)} ------------"
                )
                previous_stamp = self.global_clock
            event: simulation_event.SimulationEvent = heapq.heappop(self.events)
            event.advance_to_schedule_time(self)
            event.wrapper_process_event(self)
        all_req_outputs = [{rid: asdict(rq)} for rid, rq in self.request_output.items()]
        logging.info(
            f"Scheduling waiting overhead(s): {[r.model_rpc.schedule_waiting_overhead for r in self.runtimes]}"
            f"total schedule overhead(s): {[r.model_rpc.total_scheduling_overhead for r in self.runtimes]}"
        )
        logging.info(
            f"total recomputed tokens: {[r.model_rpc.recomputed_tokens for r in self.runtimes]}, "
            f"total forwarded tokens: {[r.model_rpc.total_forwarded_tokens for r in self.runtimes]}, "
            f"total cache hit tokens: {[r.model_rpc.total_cache_hit_tokens for r in self.runtimes]}"
        )

        with open("output.json", "w") as f:
            f.write(json.dumps(all_req_outputs, indent=4))
        return [rq for rq in self.request_output.values()]
        # return list(self.request_output.values())

    # requests: List[Dict[str, Dict, List]]
    def initialize_all_request_with_rps(
        self,
        requests,
        rps,
        time,
        send_out_times=None,
    ):
        self.time_litmit = time
        if rps != float("inf") and time != float("inf"):
            assert len(requests) >= int(rps * time)
        max_prompt_len = 0
        min_new_tokens = float("inf")
        if not send_out_times:
            send_time = self.global_clock
            for request in requests:
                max_prompt_len = max(max_prompt_len, len(request["input_ids"]))
                min_new_tokens = min(
                    min_new_tokens, request["sampling_params"]["max_new_tokens"]
                )
                self.add_event(SendRequestEvent(send_time, request))
                if rps == float("inf"):
                    interval = 0
                else:
                    interval = np.random.exponential(1 / rps)
                send_time += interval
        else:
            for request, send_time in zip(requests, send_out_times):
                max_prompt_len = max(max_prompt_len, len(request["input_ids"]))
                min_new_tokens = min(
                    min_new_tokens, request["sampling_params"]["max_new_tokens"]
                )
                self.add_event(SendRequestEvent(send_time, request))
        self.unfinished_requests = len(requests)
        logging.info(
            f"Max prompt len: {max_prompt_len}, Min new tokens: {min_new_tokens}"
        )

    def start_model_forwarding_loop(self):
        for i in range(len(self.runtimes)):
            self.add_event(ModelStepEvent(0.0, i))

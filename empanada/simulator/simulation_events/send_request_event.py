import time
import logging
from typing import TYPE_CHECKING

import numpy as np

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
)

from preble.benchmarks.benchmark_utils import RequestFuncOutput

from empanada.simulator.simulation_events.simulation_event import (
    SimulationEvent,
)
from empanada.utils.uuid import random_uuid_string
from empanada.simulator.simulation_events.generate_request_event import (
    GenerateRequestEvent,
)

if TYPE_CHECKING:
    from empanada.simulator.simulation import Simulation


class SendRequestEvent(SimulationEvent):
    """Send Request
    1. Choose the runtime to send the request to
    2. Spwan a generate request event at server side in case want to simulate network latency
    """

    def __init__(self, time: float, request):
        super().__init__("send_request", time, -1)
        self.request = request

    def advance_to_schedule_time(self, simulator: "Simulation"):
        simulator.global_clock = max(simulator.global_clock, self.time)

    def select_and_prepare_input(
        self,
        simulator: "Simulation",
        text,
        sampling_params,
        input_ids,
        rid=None,
    ):
        experiment_id = sampling_params.pop("experiment_id", random_uuid_string())
        if rid is None:
            rid = random_uuid_string()
        hit_rates = [
            r.model_rpc.get_hit_ratio() if r.server_args.report_hit_ratio else 0.0
            for r in simulator.runtimes
        ]
        highest_idx = int(np.argmax(hit_rates))
        if hit_rates[highest_idx] < 0.7:
            highest_idx = None
        # highest_idx = None
        runtime_id = simulator.router.select_runtime(
            text,
            experiment_id,
            rid,
            input_ids,
            sampling_params=sampling_params,
            current_time_stamp=self.time,
            runtime_id_with_highest_hit_rate=highest_idx,
            hit_rates=hit_rates,
        )
        generate_input = GenerateReqInput(
            text=text,
            sampling_params=sampling_params,
            rid=rid,
            stream=True,
        )
        return runtime_id, generate_input

    def process_event(self, simulator: "Simulation"):
        start = time.time()
        runtime_id, generate_input = self.select_and_prepare_input(
            simulator, **self.request
        )
        runtime = simulator.runtimes[runtime_id]
        overhead = time.time() - start
        if overhead > 0.03:
            logging.debug(f"Select runtime overhead: {overhead}")

        self.update_lock(overhead, simulator)
        simulator.request_output[generate_input.rid] = RequestFuncOutput(
            rid=generate_input.rid,
            prompt_text=self.request["text"][:20],
            prompt_len=len(self.request["input_ids"]),
            send_out_time=self.time,
            route_dest=runtime_id,
            scheduling_overhead=overhead,
            runtime_selected=runtime_id,
            max_new_tokens=self.request["sampling_params"]["max_new_tokens"],
        )
        simulator.rid_to_input[generate_input.rid] = self.request
        simulator.add_event(GenerateRequestEvent(self.time, generate_input, runtime_id))

import logging
import time
from typing import TYPE_CHECKING

from sglang.srt.managers.io_struct import (
    BatchTokenIDOut,
)
from sglang.global_config import global_config

from preble.empanada.simulator.simulation_events.simulation_event import (
    SimulationEvent,
)
from preble.empanada.simulator.server_runtime_simulator import (
    ServerRuntimeSimulator,
)

if TYPE_CHECKING:
    from preble.empanada.simulator.simulation import Simulation


class ModelStepEvent(SimulationEvent):
    def __init__(self, time: float, runtime_id: int):
        super().__init__("model_step", time, runtime_id)

    def advance_to_schedule_time(self, simulator: "Simulation"):
        runtime = simulator.runtimes[self.runtime_id]
        runtime.manager_clock = max(runtime.manager_clock, self.time)
        runtime.local_clock = max(runtime.local_clock, self.time)

    def update_metric(self, simulator: "Simulation", out_pyobjs: list[BatchTokenIDOut]):
        if out_pyobjs:
            runtime = simulator.runtimes[self.runtime_id]
            meta = "\n".join(str(obj.brief()) for obj in out_pyobjs)
            logging.debug(f"{self.runtime_id} output obj: {meta}")
            for obj in out_pyobjs:
                for rid, output_token_ids, finished in zip(
                    obj.rids, obj.output_tokens, obj.finished
                ):
                    request_output = simulator.request_output[rid]
                    if not request_output.ttft:
                        request_output.ttft = (
                            runtime.manager_clock - request_output.send_out_time
                        )
                    request_output.request_latency = (
                        runtime.manager_clock - request_output.send_out_time
                    )
                    request_output.output_len = len(output_token_ids)
                    request_output.tpot = (
                        request_output.request_latency - request_output.ttft
                    ) / request_output.output_len
                    if finished:
                        request_output.success = True
                        request_output.global_time = runtime.manager_clock
                        simulator.unfinished_requests -= 1
                        if rid in simulator.rid_to_input:
                            request = simulator.rid_to_input[rid]
                            text = request["text"]
                            input_ids = request["input_ids"]
                            simulator.router.finish_request(
                                text=text,
                                request_id=rid,
                                input_ids=input_ids,
                                experiment_id=None,
                                func_output=request_output,
                            )

    def process_event(self, simulator: "Simulation"):
        start = time.time()
        runtime = simulator.runtimes[self.runtime_id]
        next_step_input = list(runtime.manager_recv_reqs)
        runtime.manager_recv_reqs = []
        forward_time, out_pyobjs = runtime.simulate_step(next_step_input, self.time)
        sleep_time = 0.0006
        # sleep_time = 0.01
        if len(out_pyobjs) != 0:
            has_finished = any([obj.finished for obj in out_pyobjs])
            if has_finished:
                if global_config.request_dependency_time > 0:
                    sleep_time = global_config.request_dependency_time
        overhead = time.time() - start + forward_time + sleep_time
        self.update_lock(overhead, simulator, ServerRuntimeSimulator.Process.MANAGER)
        self.update_metric(simulator, out_pyobjs)
        if runtime.model_rpc.tree_cache.evicted_iteration:
            simulator.router.custom_selector.cache._update_eviction_event(
                self.runtime_id, runtime.model_rpc.tree_cache.evicted_iteration
            )
            runtime.model_rpc.tree_cache.flush_evicted()
        # logging.info(f"{self.runtime_id}: new step scheduled at manager time {runtime.manager_clock:.4f}, total {overhead:.4f}, overhead {overhead - forward_time - sleep_time:.4f}, model {forward_time:.4f}")
        simulator.add_event(ModelStepEvent(runtime.manager_clock, self.runtime_id))

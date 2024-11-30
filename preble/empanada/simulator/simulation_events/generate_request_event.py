import time
from typing import TYPE_CHECKING

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.sampling_params import SamplingParams


from preble.empanada.simulator.server_runtime_simulator import (
    ServerRuntimeSimulator,
)
from preble.empanada.simulator.simulation_events.simulation_event import (
    SimulationEvent,
)
from preble.empanada.simulator.simulation_events.add_to_manager_queue_event import (
    AddToManagerQueueEvent,
)

if TYPE_CHECKING:
    from preble.empanada.simulator.simulation import Simulation


class GenerateRequestEvent(SimulationEvent):
    """Generate Request
    1. Tokenize the input text
    2. Append to manager's waiting queue
    """

    def __init__(self, time: float, request: GenerateReqInput, runtime_id: int):
        super().__init__("generate_request", time, runtime_id)
        self.request = request

    def advance_to_schedule_time(self, simulator: "Simulation"):
        runtime = simulator.runtimes[self.runtime_id]
        runtime.tokenizer_clock = max(runtime.tokenizer_clock, self.time)
        runtime.local_clock = max(runtime.local_clock, self.time)

    def process_event(self, simulator: "Simulation"):
        start = time.time()
        simulator.request_output[self.request.rid].arrival_time = self.time
        obj = self.request
        obj.post_init()
        is_single = isinstance(obj.text, str)
        runtime = simulator.runtimes[self.runtime_id]
        tokenizer = runtime.model_rpc.tokenizer
        if is_single:
            rid = obj.rid
            # Original impl does not multi-thread this, add here to simulate tokenize latency
            input_ids = tokenizer.encode(obj.text)
            sampling_params = SamplingParams(**obj.sampling_params)
            if sampling_params.max_new_tokens != 0:
                sampling_params.normalize(tokenizer)
                sampling_params.verify()

            if isinstance(obj.image_data, list) and len(obj.image_data) > 0:
                raise NotImplementedError("Image data not supported")
            elif isinstance(obj.image_data, str):
                raise NotImplementedError("Image data not supported")
            else:
                pixel_values, image_hash, image_size = None, None, None
            tokenized_obj = TokenizedGenerateReqInput(
                rid=rid,
                input_text=obj.text,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_hash=image_hash,
                image_size=image_size,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                stream=obj.stream,
                top_logprobs_num=obj.top_logprobs_num,
                arrival_time=self.time,
            )
            overhead = time.time() - start
            self.update_lock(
                overhead, simulator, ServerRuntimeSimulator.Process.TOKENIZER
            )
            # self.update_lock(0, simulator, ServerRuntimeSimulator.Process.TOKENIZER)
            # logging.debug(f"{self.runtime_id}: tokenized req added at {runtime.tokenizer_clock}, overhead {overhead}")
            # runtime.manager_recv_reqs.append(tokenized_obj)
            simulator.add_event(
                AddToManagerQueueEvent(
                    runtime.tokenizer_clock, tokenized_obj, self.runtime_id
                )
            )
        else:
            raise NotImplementedError("Batch request not supported")

from dataclasses import dataclass
from enum import Enum
from typing import Any

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.router.model_rpc import ModelRpcServer
from sglang.srt.managers.router.model_runner import GPUConfig

from empanada.utils.uuid import random_uuid_string


@dataclass
class ServerRuntimeSimulatorParameters:
    gpu_config: GPUConfig
    server_args: ServerArgs
    profile_mode: bool


# Use simulated ModelRpcServer to maintain node state
class ServerRuntimeSimulator:
    def __init__(
        self,
        gpu_config: GPUConfig,
        server_args: ServerArgs,
        profile_mode: bool,
    ):
        self.server_args = server_args
        self.url = random_uuid_string()

        port_args = PortArgs(
            tokenizer_port=server_args.additional_ports[0],
            router_port=server_args.additional_ports[1],
            detokenizer_port=server_args.additional_ports[2],
            nccl_port=server_args.additional_ports[3],
            migrate_port=server_args.additional_ports[4],
            model_rpc_ports=server_args.additional_ports[5:],
        )
        # NOTE: some metadata is maintained in GPU memory, be careful when #replicas is too high
        self.model_rpc = ModelRpcServer(
            0, server_args, port_args, simulate=not profile_mode, gpu_config=gpu_config
        )
        self.manager_recv_reqs = []
        self.gpu_config = gpu_config
        # # Event in each queue will start from these time stamps
        # self.tokenizer_next_start_time = 0
        # self.manager_next_start_time = 0
        # self.detokenizer_next_start_time = 0
        self.local_clock = 0.0
        self.tokenizer_clock = 0.0
        self.manager_clock = 0.0

    @classmethod
    def from_server_runtime_simulator_parameters(
        cls, parameters: ServerRuntimeSimulatorParameters
    ):
        return cls(
            gpu_config=parameters.gpu_config,
            server_args=parameters.server_args,
            profile_mode=parameters.profile_mode,
        )

    def reset_clock(self):
        self.local_clock = 0.0
        self.tokenizer_clock = 0.0
        self.manager_clock = 0.0

    def simulate_step(self, recv_reqs, time) -> tuple[int, Any]:
        for recv_req in recv_reqs:
            self.model_rpc.handle_generate_request(recv_req)
        if self.model_rpc.chunk_prefill_budget > 1:
            forward_times = self.model_rpc.budget_forward_step(
                self.gpu_config.forward_simulation, time
            )
        else:
            forward_times = self.model_rpc.forward_step(
                self.gpu_config.forward_simulation, time
            )
        forward_time = sum(forward_times)
        ret = self.model_rpc.out_pyobjs
        self.model_rpc.out_pyobjs = []
        return forward_time, ret

    class Process(Enum):
        TOKENIZER = 0
        MANAGER = 1
        DETOKENIZER = 2

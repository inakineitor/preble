from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from empanada.simulator.server_runtime_simulator import (
    ServerRuntimeSimulator,
)

if TYPE_CHECKING:
    from empanada.simulator.simulation import Simulation


class SimulationEvent(ABC):
    """
    runtime_id is from external world, e.g. request generator
    """

    def __init__(self, task: str, time: float, runtime_id: int):
        self.task = task
        self.time = time
        self.runtime_id = runtime_id

    @abstractmethod
    def advance_to_schedule_time(self, simulator: Simulation):
        pass

    # call this before spawning the event
    def update_lock(
        self,
        overhead,
        simulator: Simulation,
        thread: Optional[ServerRuntimeSimulator.Process] = None,
    ):
        if self.runtime_id == -1:
            simulator.global_clock += overhead
        else:
            runtime: ServerRuntimeSimulator = simulator.runtimes[self.runtime_id]
            if thread is ServerRuntimeSimulator.Process.TOKENIZER:
                runtime.tokenizer_clock = max(runtime.tokenizer_clock, self.time)
                runtime.tokenizer_clock += overhead
            if thread is ServerRuntimeSimulator.Process.MANAGER:
                runtime.manager_clock = max(runtime.manager_clock, self.time)
                runtime.manager_clock += overhead
            runtime.local_clock = max(runtime.tokenizer_clock, runtime.manager_clock)
            simulator.global_clock = max(simulator.global_clock, runtime.local_clock)

    def wrapper_process_event(self, simulator: Simulation):
        runtime = simulator.runtimes[self.runtime_id]
        # logging.debug(f"{self.runtime_id} processing {self.task} scheduled at {self.time}, global clock: {simulator.global_clock}, local lock: {runtime.local_clock}, {runtime.tokenizer_clock}, {runtime.manager_clock}")
        self.process_event(simulator)

    @abstractmethod
    def process_event(self, simulator: Simulation):
        pass

    def __lt__(self, other):
        return self.time < other.time

    def __eq__(self, other):
        return self.time == other.time

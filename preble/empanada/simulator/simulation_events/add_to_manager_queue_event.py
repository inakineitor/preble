from typing import TYPE_CHECKING

from preble.empanada.simulator.simulation_events.simulation_event import (
    SimulationEvent,
)

if TYPE_CHECKING:
    from preble.empanada.simulator.simulation import Simulation


class AddToManagerQueueEvent(SimulationEvent):
    def __init__(self, time, tokenized_obj, runtime_id):
        super().__init__("add_to_manager_queue", time, runtime_id)
        self.tokenized_obj = tokenized_obj

    def advance_to_schedule_time(self, simulator: "Simulation"):
        pass

    def process_event(self, simulator: "Simulation"):
        runtime = simulator.runtimes[self.runtime_id]
        simulator.request_output[self.tokenized_obj.rid].append_to_queue_time = (
            self.time
        )
        runtime.manager_recv_reqs.append(self.tokenized_obj)

from dataclasses import dataclass

from empanada.simulator.simulator import SimulatorParameters


@dataclass
class ExperimentDefinition:
    experiment_name: str
    simulate: bool  # FIX: For now it only supports True
    simulator_parameters: SimulatorParameters

    def __post_init__(self):
        if not self.simulate:
            raise ValueError(
                "ExperimentDefinition only supports True as the simulate value"
            )

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from empanada.simulator.simulator import SimulatorOutput, SimulatorParameters


@dataclass(eq=True, frozen=True)
class SubexperimentDefinition:
    simulator_parameters: SimulatorParameters
    parameters: dict[str, Any] = field(hash=False)


@dataclass(eq=True, frozen=True)
class ExperimentDefinition:
    simulate: bool  # FIX: For now it only supports True
    num_replications: int
    subexperiments: list[SubexperimentDefinition]
    subexperiment_data_analyzer: Optional[
        Callable[
            [
                tuple[
                    dict[str, Any],
                    list[SimulatorOutput],
                ],
            ],
            None,
        ]
    ] = None
    joint_data_analyzer: Optional[
        Callable[
            [
                list[
                    tuple[
                        dict[str, Any],
                        list[SimulatorOutput],
                    ],
                ],
            ],
            None,
        ]
    ] = None
    override_rerun_hash_check: bool = False

    def __post_init__(self):
        if not self.simulate:
            raise ValueError(
                "ExperimentDefinition only supports True as the simulate value"
            )

    def __hash__(self):
        return hash((self.simulate, self.num_replications, tuple(self.subexperiments)))

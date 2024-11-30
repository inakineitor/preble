@dataclass
class ConfigurableMajorExperimentArgs:
    log_file_path: str
    csv_log_path: str  # for even faster parsing
    simulate: bool
    model_path: str
    experiment_type: ExperimentType
    workload_configs: List[Workload]  # Seperate policies/workloads
    experiment_name: str = "basic experiment"

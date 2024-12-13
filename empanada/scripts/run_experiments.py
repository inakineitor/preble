from empanada.experiments.experiment_runner import run_experiments


def main():
    from empanada.scripts.experiments.latency_and_ttft_across_schedulers import (
        experiments,
    )

    experiments_output = run_experiments(experiments)

    experiment_output = experiments_output[0]
    subexperiment_output = experiment_output[0]
    subexperiment_parameters, simulator_outputs = subexperiment_output
    simulator_output = simulator_outputs[0]
    requests = simulator_output.requests
    results = simulator_output.results
    request = requests[0]
    result = results[0]

    print(subexperiment_parameters)
    print(requests[0]["sampling_params"]["true_output_length"])
    print(requests[1]["sampling_params"]["true_output_length"])
    print(requests[2]["sampling_params"]["true_output_length"])
    print(result)


if __name__ == "__main__":
    main()

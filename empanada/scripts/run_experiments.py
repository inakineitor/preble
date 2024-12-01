from empanada.experiments.experiment_runner import run_experiments


def main():
    from empanada.scripts.experiments.latency_and_ttft_across_schedulers import (
        experiments,
    )

    run_experiments(experiments)


if __name__ == "__main__":
    main()

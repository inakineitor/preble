from typing import TypedDict, cast
from itertools import groupby


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_discrete

from empanada.simulator.simulator import SimulatorOutput


def unsorted_groupby(elements, key):
    return groupby(sorted(elements, key=key), key)


def calc_output_len_dist_var(
    output_length_distribution: list[tuple[float, int]]
) -> float:
    probs, vals = zip(*output_length_distribution)
    output_length_dist = rv_discrete(values=(vals, probs))
    output_length_variance = output_length_dist.var()
    return cast(float, output_length_variance)


class SubexperimentParams(TypedDict):
    num_gpus: int
    router_name: str
    requests_per_second: int
    num_workloads: int
    num_in_context_examples: int
    output_length_distribution: list[tuple[float, int]]


def num_gpus_facet_avg_norm_latency_vs_rps(
    subexperiment_outputs: list[
        tuple[
            SubexperimentParams,
            list[SimulatorOutput],
        ],
    ],
):
    list_of_key_group = [
        (num_gpus, list(group))
        for (num_gpus, group) in unsorted_groupby(
            subexperiment_outputs, lambda output: output[0]["num_gpus"]
        )
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    # Create a figure and three axes
    _, axes = plt.subplots(len(list_of_key_group), 1, figsize=(8, 12), squeeze=False)

    for axes_idx, (num_gpus, num_gpus_outputs) in enumerate(list_of_key_group):
        ax = axes[axes_idx][0]
        ax.set_title(f"{num_gpus} GPUs")
        ax.set_xlabel("Requests Per Second")
        ax.set_ylabel("Average Normalized Latency")
        ax.grid(True)
        plots = []
        for router_name, router_outputs in unsorted_groupby(
            num_gpus_outputs, lambda output: output[0]["router_name"]
        ):
            rps_list = []
            avg_norm_latency_upper_list = []
            avg_norm_latency_mean_list = []
            avg_norm_latency_lower_list = []
            for parameters, simulator_outputs in sorted(
                router_outputs, key=lambda o: o[0]["requests_per_second"]
            ):
                rps = parameters["requests_per_second"]
                avg_norm_latencies = [
                    o.benchmark_metrics.avg_norm_latency for o in simulator_outputs
                ]
                rps_list.append(rps)
                avg_norm_latency_mean_list.append(np.mean(avg_norm_latencies))
                avg_norm_latency_lower_list.append(
                    np.percentile(
                        avg_norm_latencies, 16
                    )  # Assuming normal dist. one std deviation below mean
                )
                avg_norm_latency_upper_list.append(
                    np.percentile(
                        avg_norm_latencies, 84
                    )  # Assuming normal dist. one std deviation above mean
                )

            line_plot = ax.plot(
                rps_list,
                avg_norm_latency_mean_list,
                label=router_name,
                linestyle="--",
                marker="o",
            )
            fill_between_plot = ax.fill_between(
                rps_list,
                avg_norm_latency_lower_list,
                avg_norm_latency_upper_list,
                alpha=0.5,
                label=router_name,
            )
            plots.append((line_plot, fill_between_plot))

        ax.legend(
            handles=[(p[0][0], p[1]) for p in plots],
            labels=[plot[0].get_label() for (plot, _) in plots],
            handleheight=2,
        )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.savefig("num-gpus-facet-avg-norm-latency-vs-rps.pdf")
    plt.show()


def num_gpus_facet_overhead_vs_rps(
    subexperiment_outputs: list[
        tuple[
            SubexperimentParams,
            list[SimulatorOutput],
        ],
    ],
):
    list_of_key_group = [
        (num_gpus, list(group))
        for (num_gpus, group) in unsorted_groupby(
            subexperiment_outputs, lambda output: output[0]["num_gpus"]
        )
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    # Create a figure and three axes
    _, axes = plt.subplots(len(list_of_key_group), 1, figsize=(8, 12), squeeze=False)

    for axes_idx, (num_gpus, num_gpus_outputs) in enumerate(list_of_key_group):
        ax = axes[axes_idx][0]
        ax.set_title(f"{num_gpus} GPUs")
        ax.set_xlabel("Requests Per Second")
        ax.set_ylabel("Average Scheduling Overhead")
        ax.grid(True)
        plots = []
        for router_name, router_outputs in unsorted_groupby(
            num_gpus_outputs, lambda output: output[0]["router_name"]
        ):
            rps_list = []
            avg_scheduling_overhead_upper_list = []
            avg_scheduling_overhead_mean_list = []
            avg_scheduling_overhead_lower_list = []
            for parameters, simulator_outputs in sorted(
                router_outputs, key=lambda o: o[0]["requests_per_second"]
            ):

                def avg_scheduling_overhead(o):
                    return np.mean(
                        [response.scheduling_overhead for response in o.results]
                    )

                rps = parameters["requests_per_second"]
                avg_scheduling_overheads = [
                    avg_scheduling_overhead(o) for o in simulator_outputs
                ]
                rps_list.append(rps)
                avg_scheduling_overhead_mean_list.append(
                    np.mean(avg_scheduling_overheads)
                )
                avg_scheduling_overhead_lower_list.append(
                    np.percentile(
                        avg_scheduling_overheads, 16
                    )  # Assuming normal dist. one std deviation below mean
                )
                avg_scheduling_overhead_upper_list.append(
                    np.percentile(
                        avg_scheduling_overheads, 84
                    )  # Assuming normal dist. one std deviation above mean
                )

            line_plot = ax.plot(
                rps_list,
                avg_scheduling_overhead_mean_list,
                label=router_name,
                linestyle="--",
                marker="o",
            )
            fill_between_plot = ax.fill_between(
                rps_list,
                avg_scheduling_overhead_lower_list,
                avg_scheduling_overhead_upper_list,
                alpha=0.5,
                label=router_name,
            )
            plots.append((line_plot, fill_between_plot))

        ax.legend(
            handles=[(p[0][0], p[1]) for p in plots],
            labels=[plot[0].get_label() for (plot, _) in plots],
            handleheight=2,
        )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.savefig("num_gpus_facet_overhead_vs_rps.pdf")
    plt.show()


def rps_facet_avg_norm_latency_vs_num_gpus(
    subexperiment_outputs: list[
        tuple[
            SubexperimentParams,
            list[SimulatorOutput],
        ],
    ],
):
    # print(subexperiment_outputs[0])
    list_of_key_group = [
        (requests_per_second, list(group))
        for (requests_per_second, group) in unsorted_groupby(
            subexperiment_outputs, lambda output: output[0]["requests_per_second"]
        )
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    # Create a figure and three axes
    _, axes = plt.subplots(len(list_of_key_group), 1, figsize=(8, 12), squeeze=False)

    for axes_idx, (requests_per_second, requests_per_seconds_outputs) in enumerate(
        list_of_key_group
    ):
        ax = axes[axes_idx][0]
        ax.set_title(f"{requests_per_second} Req/s")
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Average Normalized Latency")
        ax.grid(True)
        plots = []
        for router_name, router_outputs in unsorted_groupby(
            requests_per_seconds_outputs, lambda output: output[0]["router_name"]
        ):
            num_gpus_list = []
            avg_norm_latency_upper_list = []
            avg_norm_latency_mean_list = []
            avg_norm_latency_lower_list = []
            for parameters, simulator_outputs in sorted(
                router_outputs, key=lambda o: o[0]["num_gpus"]
            ):
                num_gpus = parameters["num_gpus"]
                avg_norm_latencies = [
                    o.benchmark_metrics.avg_norm_latency for o in simulator_outputs
                ]
                num_gpus_list.append(num_gpus)
                avg_norm_latency_mean_list.append(np.mean(avg_norm_latencies))
                avg_norm_latency_lower_list.append(
                    np.percentile(
                        avg_norm_latencies, 16
                    )  # Assuming normal dist. one std deviation below mean
                )
                avg_norm_latency_upper_list.append(
                    np.percentile(
                        avg_norm_latencies, 84
                    )  # Assuming normal dist. one std deviation above mean
                )

            line_plot = ax.plot(
                num_gpus_list,
                avg_norm_latency_mean_list,
                label=router_name,
                linestyle="--",
                marker="o",
            )
            fill_between_plot = ax.fill_between(
                num_gpus_list,
                avg_norm_latency_lower_list,
                avg_norm_latency_upper_list,
                alpha=0.5,
                label=router_name,
            )
            plots.append((line_plot, fill_between_plot))

        ax.legend(
            handles=[(p[0][0], p[1]) for p in plots],
            labels=[plot[0].get_label() for (plot, _) in plots],
            handleheight=2,
        )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()


# PLOT: The facet is the number of GPUs. The x-axis is the the variance of the distribution and the y-axis is the average normalized latency.
def rps_num_gpus_facet_avg_norm_latency_vs_output_length_variance(
    subexperiment_outputs: list[
        tuple[
            SubexperimentParams,
            list[SimulatorOutput],
        ],
    ],
):
    facets = [
        (facet_id, list(group))
        for (facet_id, group) in unsorted_groupby(
            subexperiment_outputs,
            lambda output: (output[0]["requests_per_second"], output[0]["num_gpus"]),
        )
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    vertically_grouped_facets = [
        (facet_id, list(group))
        for (facet_id, group) in unsorted_groupby(facets, lambda facet: facet[0][0])
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    num_vertical_facets = len(vertically_grouped_facets)
    num_horizontal_facets = (
        len(vertically_grouped_facets[0][1]) if num_vertical_facets >= 1 else 0
    )

    # Create a figure and three axes
    _, axes = plt.subplots(
        num_vertical_facets, num_horizontal_facets, figsize=(12, 12), squeeze=False
    )

    for vertical_axes_idx, (_, horizontal_facets) in enumerate(
        vertically_grouped_facets
    ):
        for horizontal_axes_idx, ((rps, num_gpus), outputs) in enumerate(
            horizontal_facets
        ):
            ax = axes[vertical_axes_idx][horizontal_axes_idx]
            ax.set_title(f"{num_gpus} GPUs with {rps} Req/s")
            ax.set_xlabel("Output Length Variance")
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
            ax.set_ylabel("Average Normalized Latency")
            ax.grid(True)
            plots = []
            for router_name, router_outputs in unsorted_groupby(
                outputs, lambda output: output[0]["router_name"]
            ):
                output_length_variance_list = []
                avg_norm_latency_upper_list = []
                avg_norm_latency_mean_list = []
                avg_norm_latency_lower_list = []
                for parameters, simulator_outputs in sorted(
                    router_outputs,
                    key=lambda o: calc_output_len_dist_var(
                        o[0]["output_length_distribution"]
                    ),
                ):
                    output_length_distribution = parameters[
                        "output_length_distribution"
                    ]
                    probs, vals = zip(*output_length_distribution)
                    output_length_dist = rv_discrete(values=(vals, probs))
                    output_length_variance = output_length_dist.var()

                    output_length_variance_list.append(output_length_variance)

                    avg_norm_latencies = [
                        o.benchmark_metrics.avg_norm_latency for o in simulator_outputs
                    ]
                    avg_norm_latency_mean_list.append(np.mean(avg_norm_latencies))
                    avg_norm_latency_lower_list.append(
                        np.percentile(
                            avg_norm_latencies, 16
                        )  # Assuming normal dist. one std deviation below mean
                    )
                    avg_norm_latency_upper_list.append(
                        np.percentile(
                            avg_norm_latencies, 84
                        )  # Assuming normal dist. one std deviation above mean
                    )

                line_plot = ax.plot(
                    output_length_variance_list,
                    avg_norm_latency_mean_list,
                    label=router_name,
                    linestyle="--",
                    marker="o",
                )
                fill_between_plot = ax.fill_between(
                    output_length_variance_list,
                    avg_norm_latency_lower_list,
                    avg_norm_latency_upper_list,
                    alpha=0.5,
                    label=router_name,
                )
                plots.append((line_plot, fill_between_plot))

            ax.legend(
                handles=[(p[0][0], p[1]) for p in plots],
                labels=[plot[0].get_label() for (plot, _) in plots],
                handleheight=2,
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.savefig("num-gpus-facet-avg-norm-latency-vs-output-length-variance.pdf")
    plt.show()


def rps_num_gpus_facet_avg_norm_latency_vs_num_workloads(
    subexperiment_outputs: list[
        tuple[
            SubexperimentParams,
            list[SimulatorOutput],
        ],
    ],
):
    facets = [
        (facet_id, list(group))
        for (facet_id, group) in unsorted_groupby(
            subexperiment_outputs,
            lambda output: (output[0]["requests_per_second"], output[0]["num_gpus"]),
        )
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    vertically_grouped_facets = [
        (facet_id, list(group))
        for (facet_id, group) in unsorted_groupby(facets, lambda facet: facet[0][0])
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    num_vertical_facets = len(vertically_grouped_facets)
    num_horizontal_facets = (
        len(vertically_grouped_facets[0][1]) if num_vertical_facets >= 1 else 0
    )

    # Create a figure and three axes
    _, axes = plt.subplots(
        num_vertical_facets, num_horizontal_facets, figsize=(12, 12), squeeze=False
    )

    for vertical_axes_idx, (_, horizontal_facets) in enumerate(
        vertically_grouped_facets
    ):
        for horizontal_axes_idx, ((rps, num_gpus), outputs) in enumerate(
            horizontal_facets
        ):
            ax = axes[vertical_axes_idx][horizontal_axes_idx]
            ax.set_title(f"{num_gpus} GPUs with {rps} Req/s")
            ax.set_xlabel("Number of ReAct Workloads")
            ax.set_ylabel("Average Normalized Latency")
            ax.grid(True)
            plots = []
            for router_name, router_outputs in unsorted_groupby(
                outputs, lambda output: output[0]["router_name"]
            ):
                num_workloads_list = []
                avg_norm_latency_upper_list = []
                avg_norm_latency_mean_list = []
                avg_norm_latency_lower_list = []
                for parameters, simulator_outputs in sorted(
                    router_outputs, key=lambda o: o[0]["num_workloads"]
                ):
                    num_workloads = parameters["num_workloads"]
                    num_workloads_list.append(num_workloads)

                    avg_norm_latencies = [
                        o.benchmark_metrics.avg_norm_latency for o in simulator_outputs
                    ]
                    avg_norm_latency_mean_list.append(np.mean(avg_norm_latencies))
                    avg_norm_latency_lower_list.append(
                        np.percentile(
                            avg_norm_latencies, 16
                        )  # Assuming normal dist. one std deviation below mean
                    )
                    avg_norm_latency_upper_list.append(
                        np.percentile(
                            avg_norm_latencies, 84
                        )  # Assuming normal dist. one std deviation above mean
                    )

                line_plot = ax.plot(
                    num_workloads_list,
                    avg_norm_latency_mean_list,
                    label=router_name,
                    linestyle="--",
                    marker="o",
                )
                fill_between_plot = ax.fill_between(
                    num_workloads_list,
                    avg_norm_latency_lower_list,
                    avg_norm_latency_upper_list,
                    alpha=0.5,
                    label=router_name,
                )
                plots.append((line_plot, fill_between_plot))

            ax.legend(
                handles=[(p[0][0], p[1]) for p in plots],
                labels=[plot[0].get_label() for (plot, _) in plots],
                handleheight=2,
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.savefig("rps-num-gpus-facet-avg-norm-latency-vs-num-workloads.pdf")
    plt.show()


def output_length_distribution_num_gpus_facet_avg_norm_latency_vs_rps(
    subexperiment_outputs: list[
        tuple[
            SubexperimentParams,
            list[SimulatorOutput],
        ],
    ],
):
    facets = [
        (facet_id, list(group))
        for (facet_id, group) in unsorted_groupby(
            subexperiment_outputs,
            lambda output: (
                output[0]["output_length_distribution"],
                output[0]["num_gpus"],
            ),
        )
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    vertically_grouped_facets = [
        (facet_id, list(group))
        for (facet_id, group) in unsorted_groupby(facets, lambda facet: facet[0][0])
    ]  # Manual conversion to list is required because list() conversion consumes grouper and they appear empty later on

    num_vertical_facets = len(vertically_grouped_facets)
    num_horizontal_facets = (
        len(vertically_grouped_facets[0][1]) if num_vertical_facets >= 1 else 0
    )

    # Create a figure and three axes
    _, axes = plt.subplots(
        num_vertical_facets, num_horizontal_facets, figsize=(12, 12), squeeze=False
    )

    for vertical_axes_idx, (_, horizontal_facets) in enumerate(
        vertically_grouped_facets
    ):
        for horizontal_axes_idx, (
            (output_length_distribution, num_gpus),
            outputs,
        ) in enumerate(horizontal_facets):
            ax = axes[vertical_axes_idx][horizontal_axes_idx]
            ax.set_title(
                f"{num_gpus} GPUs with output length distribution {output_length_distribution}"
            )
            ax.set_xlabel("Requests per Second")
            ax.set_ylabel("Average Normalized Latency")
            ax.grid(True)
            plots = []
            for router_name, router_outputs in unsorted_groupby(
                outputs, lambda output: output[0]["router_name"]
            ):
                rps_list = []
                avg_norm_latency_upper_list = []
                avg_norm_latency_mean_list = []
                avg_norm_latency_lower_list = []
                for parameters, simulator_outputs in sorted(
                    router_outputs, key=lambda o: o[0]["requests_per_second"]
                ):
                    rps = parameters["requests_per_second"]
                    rps_list.append(rps)

                    avg_norm_latencies = [
                        o.benchmark_metrics.avg_norm_latency for o in simulator_outputs
                    ]
                    avg_norm_latency_mean_list.append(np.mean(avg_norm_latencies))
                    avg_norm_latency_lower_list.append(
                        np.percentile(
                            avg_norm_latencies, 16
                        )  # Assuming normal dist. one std deviation below mean
                    )
                    avg_norm_latency_upper_list.append(
                        np.percentile(
                            avg_norm_latencies, 84
                        )  # Assuming normal dist. one std deviation above mean
                    )

                line_plot = ax.plot(
                    rps_list,
                    avg_norm_latency_mean_list,
                    label=router_name,
                    linestyle="--",
                    marker="o",
                )
                fill_between_plot = ax.fill_between(
                    rps_list,
                    avg_norm_latency_lower_list,
                    avg_norm_latency_upper_list,
                    alpha=0.5,
                    label=router_name,
                )
                plots.append((line_plot, fill_between_plot))

            ax.legend(
                handles=[(p[0][0], p[1]) for p in plots],
                labels=[plot[0].get_label() for (plot, _) in plots],
                handleheight=2,
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.savefig("output-length-distribution-num-gpus-facet-avg-norm-latency-vs-rps.pdf")
    plt.show()

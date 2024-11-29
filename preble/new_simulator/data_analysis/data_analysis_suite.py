import matplotlib.pyplot as plt
import numpy as np

from preble.benchmarks.benchmark_utils import RequestFuncOutput


def plot_latency_vs_length(
    ax: plt.Axes, max_new_tokens: list[int], request_latencies: list[float]
):
    ax.grid(True)
    ax.scatter(max_new_tokens, request_latencies, marker="x", label="Latency")
    ax.set_xlabel("Max New Tokens")
    ax.set_ylabel("Request Latency (seconds)")
    ax.set_title("Latency vs Max New Tokens")
    ax.legend()


def plot_ttft_vs_length(ax: plt.Axes, max_new_tokens: list[int], ttfts: list[float]):
    ax.grid(True)
    ax.scatter(max_new_tokens, ttfts, marker="x", color="orange", label="TTFT")
    ax.set_xlabel("Max New Tokens")
    ax.set_ylabel("TTFT (seconds)")
    ax.set_title("TTFT vs Max New Tokens")
    ax.legend()


def plot_latency_distribution(ax: plt.Axes, request_latencies: list[float]):
    # Latency distribution
    ax.hist(
        request_latencies,
        bins=np.linspace(min(request_latencies), max(request_latencies), 10),
        alpha=0.7,
        color="blue",
    )
    ax.set_xlabel("Request Latency (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title("Request Latency Distribution")


def plot_ttft_distribution(ax: plt.Axes, ttfts: list[float]):
    # TTFT distribution
    ax.hist(
        ttfts, bins=np.linspace(min(ttfts), max(ttfts), 10), alpha=0.7, color="orange"
    )
    ax.set_xlabel("TTFT (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title("TTFT Distribution")


def run_data_analysis_suite(requests: list[RequestFuncOutput]):
    # Extract data for plotting
    max_new_tokens = [req.max_new_tokens for req in requests]
    request_latencies = [req.request_latency for req in requests]
    ttfts = [req.ttft for req in requests]

    # Set 1: Latency and TTFT vs. max_new_tokens
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_latency_vs_length(ax1, max_new_tokens, request_latencies)
    plot_ttft_vs_length(ax2, max_new_tokens, ttfts)
    fig1.tight_layout()
    fig1.savefig("latency_and_ttft_vs_length.png")
    fig1.show()

    # Set 2: Latency and TTFT distributions (frequencies)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))
    plot_latency_distribution(ax3, request_latencies)
    plot_ttft_distribution(ax4, ttfts)
    fig2.tight_layout()
    fig2.savefig("latency_and_ttft_distribution.png")
    fig2.show()

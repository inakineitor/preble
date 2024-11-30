import random
import uuid
import math
import sys
import os
from typing import Optional, Union

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import copy
import re
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmarks.benchmark_workload_gen import (
    DataLoader,
    get_react_workload,
    LoadDistribution,
)


def sample_from_distribution(value_probabilities: list[tuple[float, int]]):
    probabilities, values = tuple(zip(*value_probabilities))
    random_generator = np.random.default_rng()
    return int(random_generator.choice(values, p=probabilities))


class HighVarianceWorkloadPrefixDataLoader(DataLoader):
    def __init__(
        self,
        num_patterns: int,
        total_num_requests: int,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        load_dist: LoadDistribution = LoadDistribution.EVEN,
        distribution_of_non_shared: float = 0.0,
        # output_len: int = 1,
        output_length_distribution: list[tuple[float, int]] = [
            (1.0, 1)
        ],  # List of (probablity, output length)
        num_in_context_examples: int = 4,
        random_workload_path=None,
        workload_start_from: int = 0,
        decoding_size=None,
        context_len: Optional[
            int
        ] = None,  # if this is not none, ignore num_in_context_exampels
    ):
        super().__init__(
            "random", num_patterns, total_num_requests, tokenizer, load_dist
        )
        self.distribution_of_non_shared = distribution_of_non_shared
        # self.output_len = output_len
        self.output_length_distribution = output_length_distribution
        self.num_in_context_examples = num_in_context_examples
        self.random_workload_path = random_workload_path
        self.workload_start_from = workload_start_from
        self.decoding_size = decoding_size
        self.context_len = context_len
        if self.context_len:
            self.num_in_context_examples = math.ceil(self.context_len / 475)

    def generate_workload(self, k: int):
        num_prefixed_shared = int(
            self.total_num_requests * (1 - self.distribution_of_non_shared)
        )
        num_non_shared = int(self.total_num_requests * self.distribution_of_non_shared)
        workload = []

        def get_sampling_params():
            output_length = sample_from_distribution(self.output_length_distribution)
            return {
                "experiment_id": f"random_experiment_{self.num_patterns}_{self.distribution_of_non_shared}_{self.total_num_requests}",
                "temperature": 0,
                "max_new_tokens": output_length,
                "ignore_eos": True,  # For better micro-benchmark
            }

        for i in range(num_prefixed_shared):
            workload_num = self.workload_start_from + i % self.num_patterns
            prompt = get_react_workload(
                f"Workload {workload_num} ", num_examples=self.num_in_context_examples
            )
            workload.append(
                {
                    "text": prompt,
                    "sampling_params": get_sampling_params(),
                    "rid": uuid.uuid4().hex,
                }
            )

        # random_workload = generate_random_workload(random_workload_path=self.random_workload_path)
        for _ in range(num_non_shared):
            # prompt = random.choice(random_workload)
            prompt = get_react_workload(
                uuid.uuid4().hex + " ", num_examples=self.num_in_context_examples
            )
            workload.append(
                {
                    "text": prompt,
                    "sampling_params": get_sampling_params(),
                    "rid": uuid.uuid4().hex,
                }
            )
        self.add_input_token_ids_to_workload(workload)

        random.shuffle(workload)
        # prompt_lens = [len(p["input_ids"]) for p in workload]
        # plt.hist(prompt_lens)
        # plt.savefig(f"react_prompt_length.png")
        return workload

    @staticmethod
    def is_hot(output):
        return output.prompt_text.startswith("Workload ")

    @staticmethod
    def get_prefix_index(output):
        match = re.search(r"\bWorkload\s+(\d+)", output.prompt_text)
        if match:
            return int(match.group(1))
        else:
            return None

    def workload_specific_args(self):
        return {
            "num_patterns": self.num_patterns,
            "total_num_requests": self.total_num_requests,
            "load_dist": str(self.load_dist),
            "random_ratio": self.distribution_of_non_shared,
            "output_length_distribution": self.output_length_distribution,
            "num_in_context_examples": self.num_in_context_examples,
        }

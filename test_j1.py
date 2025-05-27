import argparse
import backoff
import textwrap
from datasets import load_dataset
from pydantic import Field
from pydantic.fields import FieldInfo
from verdict.common.judge import JudgeUnit
from verdict.core.pipeline import Pipeline
from verdict.extractor import RawExtractor
from verdict.model import vLLMModel
from verdict.schema import Schema
from verdict.scale import Scale, ScaleType
from verdict.util.ratelimit import (
    RateLimitPolicy,
    ConcurrentRateLimiter,
)
from typing import Dict, List, Tuple, Any
from rewards import extract_scores
from utils import judge_prompt_template, judge_system_prompt


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    max_time=30,
)
def run_pipeline_with_retry(pipeline: Pipeline, inputs: List[Schema]):
    return pipeline.run_from_list(inputs, graceful=True, display=True)


def try_extract_scores(row, columns):
    try:
        return extract_scores(row[columns[-1]])
    except Exception:
        return "Output does not match j1 format"


class IdentityScale(Scale):
    def __init__(self) -> None:
        super().__init__(ScaleType.DISCRETE, end_is_worst=False)
        self.T = str

    def pydantic_fields(self, key: str = "output") -> Dict[str, Tuple[Any, FieldInfo]]:
        return {
            key: (str, Field(...)),
        }

    def prompt(self) -> str:
        return ""

    def __str__(self) -> str:
        return "IdentityScale"


def main(
    api_base: str,
    model_name: str,
    output_file: str,
    max_seq_length: int = 6144,
    max_prompt_length: int = 2048,
    retries: int = 3,
):

    j1_micro = vLLMModel(
        name=model_name,
        api_base=api_base,
        api_key="token-abc123",
        rate_limiter=RateLimitPolicy.using(
            requests=ConcurrentRateLimiter(max_concurrent=10),
        ),
    )

    judge = JudgeUnit(scale=IdentityScale())
    instruction_prompt = (
        judge_prompt_template.replace("conversation_context_query", "source.prompt")
        .replace("response_a", "source.chosen")
        .replace("response_b", "source.rejected")
    )

    pipeline = Pipeline() >> judge.prompt(
        f"@system\n{judge_system_prompt()}\n\n@user\n{instruction_prompt}"
    ).extract(RawExtractor()).via(
        j1_micro,
        retries=retries,
        max_seq_length=max_seq_length,
        max_prompt_length=max_prompt_length,
    )

    ds = load_dataset("allenai/reward-bench")["filtered"]
    prompts = ds["prompt"]
    chosens = ds["chosen"]
    rejecteds = ds["rejected"]

    input_schemas = [
        Schema.of(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
        )
        for prompt, chosen, rejected in zip(prompts, chosens, rejecteds)
    ]

    result_df, columns = run_pipeline_with_retry(pipeline, input_schemas)
    result_df["pairwise_score"] = result_df.apply(
        lambda row: try_extract_scores(row, columns), axis=1
    )
    result_df["correct"] = result_df.apply(
        lambda row: row["pairwise_score"][0] > row["pairwise_score"][1], axis=1
    )
    result_df.to_csv(output_file, index=False)
    print(
        f"{model_name} RewardBench Accuracy: ",
        sum(result_df["correct"]) / len(result_df),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1"
    )
    parser.add_argument("--model-name", type=str, default="j1-micro")
    parser.add_argument("--output-file", type=str, default="results/j1_rewardbench.csv")
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    main(
        args.api_base,
        args.model_name,
        args.output_file,
        args.max_seq_length,
        args.max_prompt_length,
        args.retries,
    )

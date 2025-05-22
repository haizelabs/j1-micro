import re
from config import Config
from typing import List


class FormatError(Exception):
    pass


def extract_spct_scores(raw_response: str) -> List[float]:
    """
    Extract the SPCT scores from the raw response.
    Expects the following format:
        ---
        <specific_criteria>...</specific_criteria>
        <analysis>...</analysis>
        <scores>\boxed{x, y}</scores>
        ---
    """
    match = re.search(r"<scores>(.*?)</scores>", raw_response, re.DOTALL)
    if not match:
        raise FormatError("No SPCT scores found in response")

    boxed_match = re.search(r"\\boxed{([\d.]+),\s*([\d.]+)}", match.group(1))
    if not boxed_match:
        raise FormatError("No boxed scores found in scores tag")

    try:
        return [float(boxed_match.group(1)), float(boxed_match.group(2))]
    except ValueError:
        raise FormatError("Invalid score format in boxed response")


def spct_format_reward_func(completions: List[dict[str, str]], **kwargs) -> List[float]:
    """
    SPCT format reward for the n=2 (pairwise preference) case

    > Range: [0, 0.2]
    > Boxed contributes 1/4 of the total score
    > Each of the 6 tags contributes 1/8 of the total score
    """
    scores = []
    for completion in completions:
        boxed_fmted = True
        raw_response = completion[0]["content"]
        try:
            extract_spct_scores(raw_response)
        except FormatError:
            boxed_fmted = False
        finally:
            required_tags = [
                "<specific_criteria>",
                "</specific_criteria>",
                "<analysis>",
                "</analysis>",
                "<scores>",
                "</scores>",
            ]
            denominator = len(required_tags) + 2
            total_score = 0.0
            for tag in required_tags:
                if tag in raw_response:
                    total_score += 1.0 / denominator

            if boxed_fmted:
                total_score += 2.0 / denominator

            scores.append(total_score / 5)

    return scores


def spct_argmax_reward_func(
    completions: List[dict[str, str]], chosen_positions: List[str], **kwargs
) -> List[float]:
    """
    For the n=2 (pairwise preference) case, the `chosen` response should score higher than the `rejected` response.

    > Range: [0, 1]
    > completions: list of rollouts
    > chosen_positions: list of positions (A or B) of the `chosen` response; see `spct_judge_prompt_format
    """
    scores = []
    for completion, chosen_position in zip(completions, chosen_positions):
        print("\n" + "-" * 50 + " Argmax Reward " + "-" * 50 + "\n")
        raw_response = completion[0]["content"]
        try:
            extracted_score_box = extract_spct_scores(raw_response)
        except FormatError:
            scores.append(0.0)
            continue
        else:
            position_to_score = {
                Config.CHOSEN_POSITION_A: lambda scores: scores[0] > scores[1],
                Config.CHOSEN_POSITION_B: lambda scores: scores[1] > scores[0],
            }
            if chosen_position not in position_to_score:
                raise ValueError(f"Invalid chosen position: {chosen_position}")
            scores.append(
                1.0 if position_to_score[chosen_position](extracted_score_box) else 0.0
            )

    return scores

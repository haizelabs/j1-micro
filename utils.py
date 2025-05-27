import os
import pandas as pd
import textwrap
from config import Config
from datasets import Dataset, load_dataset
from random import random


def judge_system_prompt() -> str:
    return textwrap.dedent(
        """
    You are an expert XML wrangler. You must respond in the following format, regardless of the input:
    
    <specific_criteria>
    ...
    </specific_criteria>
    <analysis>
    ...
    </analysis>
    <scores>
    \\boxed{{..., ...}}
    </scores>

    Please only respond in English.
    """
    ).strip()


judge_prompt_template = textwrap.dedent(
    """
    You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
    Given the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them.
    Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
    Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

    #### Evaluation Criteria ####
    1. Instruction Adherence:
    - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.
    - Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.
    - Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.
    - Not Adhered (1-2 points): The response does not meet any instructions.
    Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered."
    2. Usefulness:
    - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.
    - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.
    - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.
    - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.
    Example: If there are factual errors in the response but the overall direction is correct, it falls under "Useful but Incomplete."
    3. Level of Detail:
    - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.
    - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.
    - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.
    - Not Detailed (1-2 points): The response is very brief and lacks necessary details.
    Example: If the response provides only a simple conclusion without an explanation, it falls under "Not Detailed."
    4. Relevance:
    - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.
    - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.
    - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.
    - Not Relevant (1-2 points): The response is completely irrelevant.
    Example: If the response strays from the topic but still provides some relevant information, it falls under "Partially Relevant."

    #### Conversation Context ####
    {conversation_context_query}
    #### Responses to be Scored ####
    [The Begin of Response A]
    {response_a}
    [The End of Response A]
    [The Begin of Response B]
    {response_b}
    [The End of Response B]
    #### Output Format Requirements ####

    Output with three lines
    <specific_criteria>
    [Other potential criteria specific to the query and the context, and the weights of each criteria.]
    </specific_criteria>
    <analysis>
    [Compare different responses based on given Criteria.]
    </analysis>
    <scores>
    [The overall comprehensive score of all responses in order, separate by comma in the boxed, e.g., \\boxed{{x, x}} if there exists 2 responses.]
    </scores>
    """
).strip()


def judge_prompt_format(
    conversation_context_query: str, response_a: str, response_b: str
) -> str:
    """
    See page 40 of https://arxiv.org/abs/2504.02495
    """

    return judge_prompt_template.format(
        conversation_context_query=conversation_context_query,
        response_a=response_a,
        response_b=response_b,
    )


def format_inputs(
    row: dict[str, str], use_system_prompt: bool = True
) -> list[dict[str, str]]:
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": judge_system_prompt()})
    input = row[Config.COLUMN_INPUT]
    if row[Config.COLUMN_CHOSEN_POSITION] == Config.CHOSEN_POSITION_A:
        response_a = row[Config.COLUMN_CHOSEN]
        response_b = row[Config.COLUMN_REJECTED]
    else:
        response_a = row[Config.COLUMN_REJECTED]
        response_b = row[Config.COLUMN_CHOSEN]
    judge_prompt = judge_prompt_format(
        conversation_context_query=input,
        response_a=response_a,
        response_b=response_b,
    )
    messages.append({"role": "user", "content": judge_prompt})
    return messages


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the dataframe into TRL-friendly format.
    """
    df[Config.COLUMN_INPUT] = df[Config.COLUMN_CHOSEN_ORIG].apply(
        lambda x: x[0]["content"]
    )
    df[Config.COLUMN_CHOSEN] = df[Config.COLUMN_CHOSEN_ORIG].apply(
        lambda x: x[1]["content"]
    )
    df[Config.COLUMN_REJECTED] = df[Config.COLUMN_REJECTED_ORIG].apply(
        lambda x: x[1]["content"]
    )
    df[Config.COLUMN_CHOSEN_POSITION] = [
        Config.CHOSEN_POSITION_A if random() < 0.5 else Config.CHOSEN_POSITION_B
        for _ in range(len(df))
    ]
    return df


def get_skywork_dataset(
    file_name: str,
    split: str = "train",
    ds_name: str = "Skywork/Skywork-Reward-Preference-80K-v0.2",
) -> Dataset:
    """
    See https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2
    """

    assert split in [
        "train",
        "test",
    ], f"Invalid `split` argument: {split}. Expected 'train' or 'test'."

    assert (
        ".csv" in file_name
    ), f"Invalid `file_name` argument: {file_name}. Expected a .csv file."

    if os.path.exists(file_name):
        assert (
            split in file_name
        ), f"Invalid split `{split}` for file_name: `{file_name}`"
        df = pd.read_csv(file_name)

    else:
        ds = load_dataset(ds_name)
        ds = ds["train"].train_test_split(test_size=0.2)
        train_df = process_df(ds["train"].to_pandas())
        test_df = process_df(ds["test"].to_pandas())
        train_df.to_csv(file_name, index=False)
        test_df.to_csv(file_name, index=False)

        if split == "train":
            df = train_df
        elif split == "test":
            df = test_df

    df[Config.COLUMN_PROMPT] = df.apply(
        lambda row: format_inputs(row, use_system_prompt=True),
        axis=1,
    )
    df = df.drop(
        columns=[
            Config.COLUMN_SOURCE,
            Config.COLUMN_CHOSEN_ORIG,
            Config.COLUMN_REJECTED_ORIG,
        ]
    )
    return Dataset.from_pandas(df)

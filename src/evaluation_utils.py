# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the functions for running the experiments in the paper "Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?"."""

import functools
import gc
import itertools
import re
import string
from typing import Any

import pandas as pd
from src import model_utils
from src.data_utils import batchify
from src.model_utils import is_instruction_tuned
from src.tokenization_utils import get_completion
import torch
import tqdm
import transformers
import unidecode
from vllm import SamplingParams


# fact types for pretrained models
pretrained_fact_types = {
    "condition": "r1(e1)",
    "base": "r2(e2)",
    "entailed": "r2(r1(e1))",
    "entailed.e2.null": "r2(e2.null)",
    "entailed.e1.null": "r2(r1(e1.null))",
}

# fact types for instruction-tuned models
it_fact_types = {
    "blank.cot": "r2(r1(e1)).blank.cot",
    "blank.entailed": "r2(r1(e1)).blank",
    "blank.condition": "r1(e1).blank",
    "blank.base": "r2(e2).blank",
    "blank.e2.null": "r2(e2.null).blank",
    "blank.e1.null": "r2(r1(e1.null)).blank",
}

# answer entity types for fact types
answer_entity_types = {
    "r1(e1)": "e2",
    "r2(e2)": "e3",
    "r2(r1(e1))": "e3",
    "r2(r1(e1)).appositive": "e2",
    "r2(e2.null)": "e3",
    "r2(r1(e1.null))": "e3",
    "r2(e1)": "e3",
    "r2(r1(e1)).cot": "e3",
    "r1(e1).blank": "e2",
    "r2(e2).blank": "e3",
    "r2(r1(e1)).blank": "e3",
    "r2(r1(e1)).blank.cot": "e3",
    "r2(e2.null).blank": "e3",
    "r2(r1(e1.null)).blank": "e3",
    "r2(r1(e1)).hint_think": "e3",
}


def get_completion_messages(
    prompt: str, prompt_key: str, model_name: str
) -> list[dict[str, str]]:
  """Get the completion messages for the prompt.

  The messages are based on the prompt key and model name for instruction-tuned
  models.

  Args:
      prompt: The prompt text.
      prompt_key: The key indicating the type of prompt.
      model_name: The name of the model.

  Returns:
      A list of dictionaries containing the role and content of the messages.
  """
  if "think" in prompt_key:
    instruction = (
        "Fill in the blank. Write down only what goes in the blank. Think"
        " step-by-step, but do it only internally and do not explain it in the"
        " answer. The answer can consist of multiple words."
    )
    separator = "\n\n"
  elif "blank" in prompt_key:
    if prompt_key.endswith("cot.prompt"):
      instruction = (
          "Fill in the blank. First, write the step-by-step explanation"
          ' necessary to get the solution with the prefix "EXPLANATION:". After'
          ' that, write down the final answer with the prefix "ANSWER:". For'
          " the final answer, write down only what goes in the blank. The"
          " answer can consist of multiple words."
      )
    else:
      instruction = (
          "Fill in the blank. Write down only what goes in the blank. Do not"
          " explain your answer. The answer can consist of multiple words."
      )
    separator = "\n\n"
  else:
    raise ValueError(f"Unknown prompt key: {prompt_key}")

  start_role = model_utils.get_messages_start_role(model_name)

  if start_role == "user":
    return [
        {"role": "user", "content": f"{instruction}{separator}{prompt}"},
    ]
  else:
    return [
        {"role": start_role, "content": instruction},
        {"role": "user", "content": prompt},
    ]


def get_input_dict(
    tokenizer: Any, prompt_key: str, prompts: list[str], instruction_tuned: bool
) -> dict[str, Any]:
  """Get the input dictionary for the vLLm models based on the prompts and whether the model is instruction-tuned.

  Args:
      tokenizer: The tokenizer to use.
      prompt_key: The key indicating the type of prompt.
      prompts: The list of prompts.
      instruction_tuned: Whether the model is instruction-tuned.

  Returns:
      A dictionary containing the input prompts and sampling parameters.
  """
  input_dict = dict()
  if instruction_tuned:
    assert (
        tokenizer.chat_template
    ), "Instruction-tuned models require chat templates"
    print("Using chat template")
    messages = [
        get_completion_messages(prompt, prompt_key, tokenizer.name_or_path)
        for prompt in prompts
    ]
    input_dict["prompts"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
  else:
    input_dict["prompts"] = prompts

  input_dict["sampling_params"] = get_vllm_sampling_params(
      prompt_key, instruction_tuned
  )

  return input_dict


def get_vllm_sampling_params(
    prompt_key: str, instruction_tuned: bool
) -> SamplingParams:
  """Get the sampling parameters for the vLLM model based on the prompt key and whether the model is instruction-tuned.

  Args:
      prompt_key: The key indicating the type of prompt.
      instruction_tuned: Whether the model is instruction-tuned.

  Returns:
      The sampling parameters.
  """
  params = {"seed": 0}

  if "blank" in prompt_key:  # instruction-tuned models
    if "cot" in prompt_key:
      params["max_tokens"] = 512
    else:
      params["max_tokens"] = 96
  elif (
      instruction_tuned
  ):  # instruction-tuned models might repeat the input,
    # so we need to increase the max tokens
    params["max_tokens"] = 96
  else:
    params["max_tokens"] = 32

  params["temperature"] = 0  # greedy decoding without randomness

  return SamplingParams(**params)


def run_vllm_completion(
    llm: Any,
    df: pd.DataFrame,
    fact_type: str,
    instruction_tuned: bool,
    force_completion: bool = False,
) -> pd.DataFrame:
  """Run prompt completion with vLLM models for the given DataFrame and composition type.

  Args:
      llm: The vLLM model.
      df: The input DataFrame.
      fact_type: The fact composition type.
      instruction_tuned: Whether the model is instruction-tuned.
      force_completion: Whether to force completion even if it already exists.

  Returns:
      The updated DataFrame with completions.
  """
  if force_completion or (f"{fact_type}.completion" not in df):
    assert len(df) == len(df["uid"].unique())

    prompt_key = f"{fact_type}.prompt"

    # If there are many same prompts, group by prompt and generate
    # completions for each group
    prompt_to_uids = (
        df.groupby(prompt_key).apply(lambda x: set(x["uid"].tolist())).to_dict()
    )
    prompts = sorted(set(prompt_to_uids.keys()), key=len, reverse=True)
    outputs = llm.generate(
        **get_input_dict(
            llm.get_tokenizer(), prompt_key, prompts, instruction_tuned
        )
    )

    update_df_with_completion(df, prompts, outputs, prompt_to_uids, fact_type)

  return df


def run_hf_completion(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    fact_type: str,
    instruction_tuned: bool,
    batch_size: int = 4,
    force_completion: bool = False,
) -> pd.DataFrame:
  """Run prompt completion with HuggingFace models for the given DataFrame and composition type.

  Args:
      model: The Hugging Face model.
      tokenizer: The tokenizer to use.
      df: The input DataFrame.
      fact_type: The fact composition type.
      instruction_tuned: Whether the model is instruction-tuned.
      batch_size: The batch size for processing.
      force_completion: Whether to force completion even if it already exists.

  Returns:
      The updated DataFrame with completions.
  """
  if force_completion or (f"{fact_type}.completion" not in df):
    assert len(df) == len(df["uid"].unique())

    completions = batchify(
        get_hf_completions,
        batch_size=batch_size,
    )(
        {"prompts": df[f"{fact_type}.prompt"].tolist()},
        model=model,
        tokenizer=tokenizer,
        prompt_key=f"{fact_type}.prompt",
        instruction_tuned=instruction_tuned,
    )

    df.loc[:, f"{fact_type}.completion"] = completions

  return df


def get_hf_completions(
    model: Any,
    tokenizer: Any,
    prompt_key: str,
    prompts: list[str],
    instruction_tuned: bool,
) -> list[str]:
  """Get HuggingFace model completions for the given prompts.

  Args:
      model: The Hugging Face model.
      tokenizer: The tokenizer to use.
      prompt_key: The key indicating the type of prompt.
      prompts: The list of prompts.
      instruction_tuned: Whether the model is instruction-tuned.

  Returns:
      The list of completions.
  """
  input_dict = get_input_dict(
      tokenizer,
      prompt_key,
      prompts,
      instruction_tuned,
  )
  prompts = input_dict["prompts"]
  new_max_tokens = input_dict["sampling_params"].max_tokens

  assert tokenizer.padding_side == "left"
  prompt_inputs = tokenizer(
      prompts, return_tensors="pt", padding=True, truncation=True
  ).to(model.device)

  if "Nemo-Base" in model.name_or_path:
    del prompt_inputs["token_type_ids"]

  completions = model.generate(
      **prompt_inputs,
      pad_token_id=tokenizer.eos_token_id,
      max_new_tokens=new_max_tokens,
      do_sample=False,
  )
  return get_completion(completions, prompt_inputs, tokenizer)


def update_df_with_completion(
    df: pd.DataFrame,
    prompts: list[str],
    outputs: Any,
    prompt_to_uids: dict[str, list[int]],
    fact_type: str,
) -> None:
  """Update the DataFrame with completions.

  Args:
      df: The input DataFrame.
      prompts: The list of prompts.
      outputs: The generated outputs.
      prompt_to_uids: A dictionary mapping prompts to UIDs.
      fact_type: The fact composition type.
  """
  results = []
  for prompt, output in tqdm.tqdm(
      zip(prompts, outputs),
      total=len(prompts),
      desc=f"Extracting {fact_type} completions",
  ):
    uids = prompt_to_uids[prompt]

    completion = output.outputs[0].text
    results.extend([(uid, completion) for uid in uids])

  # Create DataFrame from results
  new_columns = [f"{fact_type}.completion"]

  for column in new_columns:
    if column in df:
      df.drop(columns=[column], inplace=True)

  results_df = pd.DataFrame(results, columns=["uid"] + new_columns)
  merged = df.merge(results_df, on="uid", how="left")

  completions = merged[f"{fact_type}.completion"].tolist()
  df.insert(
      df.columns.get_loc(f"{fact_type}.prompt") + 1,
      f"{fact_type}.completion",
      completions,
  )


def shortcut_free_evaluate(
    df: pd.DataFrame,
    fact_type: str,
    answer_entity_type: str,
    answer_postfix: str = "aliases",
    normalize: bool = True,
) -> pd.DataFrame:
  """Evaluate the completions for a fact_type in a shortcut-free manner.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
      answer_entity_type: The answer entity type.
      answer_postfix: The postfix for the answer column.
      normalize: Whether to normalize the text.

  Returns:
      The updated DataFrame with evaluation results.
  """
  matches_col = "matches" if normalize else "strict.matches"
  correct_col = "correct" if normalize else "strict.correct"
  inst_col = "failed_instruction" if normalize else "strict.failed_instruction"

  df.loc[:, f"{fact_type}.completion"] = df.loc[
      :, f"{fact_type}.completion"
  ].astype(str)

  df.loc[:, f"{fact_type}.{matches_col}"] = df.apply(
      lambda row: get_matches(
          row[f"{fact_type}.completion"],
          row[f"{answer_entity_type}.{answer_postfix}"],
          normalize=normalize,
      ),
      axis=1,
  )
  df.loc[:, f"{fact_type}.{correct_col}"] = df.loc[
      :, f"{fact_type}.{matches_col}"
  ].apply(lambda x: len(x) > 0)  # pylint: disable=g-explicit-length-test

  if fact_type.startswith(("r1(e1)", "r2(e2)")):
    fill_has_multiple_choice_format(df, fact_type)
    condition = (
        df[f"{fact_type}.{correct_col}"]
        & df[f"{fact_type}.has_multiple_choice_format"]
    )
    df.loc[condition, f"{fact_type}.{correct_col}"] = False

  if fact_type not in ["r2(r1(e1))", "r2(r1(e1)).blank"]:
    return df

  completion_col = "completion"

  # Direct assignment by expanding the tuples/lists returned by apply
  results = df.apply(
      lambda row: get_real_correct_and_failed_instruction(
          row, fact_type, completion_col, normalize=normalize
      ),
      axis=1,
  )

  # Separate results into the respective columns
  df.loc[:, f"{fact_type}.real.{correct_col}"] = results.apply(lambda x: x[0])
  df.loc[:, f"{fact_type}.{inst_col}"] = results.apply(lambda x: x[1])

  df.loc[:, f"{fact_type}.real.{correct_col}"] = df[
      f"{fact_type}.real.{correct_col}"
  ].astype(bool)
  df.loc[:, f"{fact_type}.{inst_col}"] = df[f"{fact_type}.{inst_col}"].astype(
      bool
  )

  print(f"set unusable for {fact_type}")
  fill_has_multiple_choice_format(df, fact_type)
  condition = (
      df[f"{fact_type}.{correct_col}"]
      & df[f"{fact_type}.has_multiple_choice_format"]
  )
  df.loc[condition, f"{fact_type}.real.{correct_col}"] = False
  unusable_col = f"{fact_type}.unusable"

  unusable_condition = (
      df[f"{fact_type}.has_multiple_choice_format"]
      | df[f"{fact_type}.{inst_col}"]
  ) & df[f"{fact_type}.{correct_col}"]

  df.loc[:, unusable_col] = False
  df.loc[unusable_condition, unusable_col] = True

  df.loc[:, unusable_col] = df[unusable_col].astype(bool)

  return df


def get_real_correct_and_failed_instruction(
    row: pd.Series,
    fact_type: str,
    completion_postfix: str,
    normalize: bool = True,
) -> tuple[bool, bool]:
  """Get the real correct (set to False when e2 is generated before e3) and failed instruction status for the given row.

  Args:
      row: The input row.
      fact_type: The fact composition type.
      completion_postfix: The postfix for the completion column.
      normalize: Whether to normalize the text.

  Returns:
      A tuple containing the real correct status and failed instruction status.
  """
  assert completion_postfix in ["completion", "real.completion"]

  correct_col = "correct" if normalize else "strict.correct"

  if not row[f"{fact_type}.{correct_col}"]:
    return False, False

  completion = row[f"{fact_type}.{completion_postfix}"]
  if not completion:
    return False, False

  if normalize:
    completion = normalize_text(completion)

  e2_aliases = list(set(itertools.chain.from_iterable(row["e2.aliases"])))
  e3_aliases = list(set(itertools.chain.from_iterable(row["e3.aliases"])))

  if normalize:
    e2_aliases = list(set([normalize_text(alias) for alias in e2_aliases]))
    e3_aliases = list(set([normalize_text(alias) for alias in e3_aliases]))

    e2_aliases = [alias for alias in e2_aliases if alias]
    e3_aliases = [alias for alias in e3_aliases if alias]

  e2_indices = [completion.find(e2) for e2 in e2_aliases]
  e3_indices = [completion.find(e3) for e3 in e3_aliases]

  assert e3_indices

  e2_valid_indices = [idx for idx in e2_indices if idx != -1]
  e3_valid_indices = [idx for idx in e3_indices if idx != -1]

  if not e2_valid_indices:
    return True, False

  min_e2_index = min(e2_valid_indices)
  min_e3_index = min(e3_valid_indices)

  if min_e2_index < min_e3_index:
    return False, True

  return True, False


def fill_has_multiple_choice_format(df: pd.DataFrame, fact_type: str) -> None:
  """Fill the DataFrame with a column indicating whether the completion has the form of a multiple-choice question.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
  """
  has_multiple_choice_format = df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["A.", "B.", "C."])
  )
  has_multiple_choice_format |= df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["A)", "B)", "C)"])
  )
  has_multiple_choice_format |= df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["1.", "2.", "3."])
  )
  has_multiple_choice_format |= df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["1)", "2)", "3)"])
  )
  df.loc[:, f"{fact_type}.has_multiple_choice_format"] = (
      has_multiple_choice_format
  )


def run_shortcut_free_evaluation(
    df: pd.DataFrame, normalize: bool = True, force: bool = False
) -> None:
  """Run shortcut-free evaluation on the DataFrame.

  Args:
      df: The input DataFrame.
      normalize: Whether to normalize the text.
      force: Whether to force evaluation even if it already exists.
  """
  if normalize:
    correct_col = "correct"
  else:
    correct_col = "strict.correct"

  for col in df.columns:
    if col.endswith(".real.completion"):
      continue

    if col.endswith(".completion"):
      fact_type = col.rsplit(".", 1)[0]

      if fact_type not in list(pretrained_fact_types.values()) + list(
          it_fact_types.values()
      ):
        continue

      if not force and f"{fact_type}.{correct_col}" in df:
        continue

      print(fact_type)
      answer_entity_type = answer_entity_types[fact_type]
      df = shortcut_free_evaluate(
          df, fact_type, answer_entity_type, normalize=normalize
      )


def transform_punctuation(text: str) -> str:
  """Replace punctuation in the text with spaces.

  Args:
      text: The input text.

  Returns:
      The text with punctuation replaced by spaces.
  """
  translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
  text = text.translate(translator)
  return text


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_accents: bool = True,
    remove_articles: bool = True,
    remove_spaces_in_abbr: bool = True,
    remove_punctuations: bool = True,
    remove_spaces: bool = False,
) -> str:
  """Normalize the input text.

  Args:
      text: The input text.
      lowercase: Whether to convert the text to lowercase.
      remove_accents: Whether to remove accents from the text.
      remove_articles: Whether to remove articles from the text.
      remove_spaces_in_abbr: Whether to remove spaces in abbreviations.
      remove_punctuations: Whether to remove punctuations from the text.
      remove_spaces: Whether to remove all spaces from the text.

  Returns:
      The normalized text.
  """
  text = text.strip()

  if remove_spaces_in_abbr:
    text = re.sub(r"(?<=\b[A-Z])\. (?=[A-Z]\.)", ".", text)

  if lowercase:
    text = text.lower()

  # remove accents
  if remove_accents:
    text = unidecode.unidecode(text)

  # remove articles
  if remove_articles:
    text = re.sub(r"\b(the|an|a)\b(?=\s)", "", text, flags=re.IGNORECASE)

  # replace punctuation with spaces
  if remove_punctuations:
    text = transform_punctuation(text)

  # replace multiple spaces with single space
  text = re.sub(r"\s+", " ", text)

  # remove all spaces
  if remove_spaces:
    text = text.replace(" ", "")

  return text.strip()


def get_matches(
    completion: str, correct_answers: list[list[str]], normalize: bool = True
) -> tuple[str, ...]:
  """Get a subset of correct_answers that appear in the completion.

  Args:
      completion: The generated completion.
      correct_answers: The list of correct answers.
      normalize: Whether to normalize the text.

  Returns:
      A tuple of the matched answers that appear in the completion.
  """
  assert isinstance(correct_answers, (list, tuple)), correct_answers
  if correct_answers:
    assert isinstance(correct_answers[0], (list, tuple)), correct_answers[0]

  completion = completion.strip()
  if normalize:
    completion = normalize_text(completion)

  matched_answers = []
  for answers in correct_answers:
    for possible_answer in answers:
      possible_answer = f"{possible_answer}"

      possible_answer_to_compare = possible_answer.strip()
      if normalize:
        possible_answer_to_compare = normalize_text(possible_answer_to_compare)

      if not possible_answer_to_compare:
        continue

      pattern = r"\b" + re.escape(possible_answer_to_compare) + r"\b"
      if re.search(pattern, completion):
        matched_answers.append(possible_answer)

  if not matched_answers:
    return ()
  return tuple(matched_answers)


def get_df_with_shortcut_free_metrics(
    df: pd.DataFrame, blank_cols_already_processed: bool = False
) -> pd.DataFrame:
  """Get the DataFrame with the information that can calculate the shortcut-free evaluation metric of latent composability.

  The metric can be calculated by the following formula:
  composability = sum(composability_numer) / sum(composability_denom)

  Args:
      df: The input DataFrame.
      blank_cols_already_processed: Whether the blank columns are already
        processed.

  Returns:
      The updated DataFrame with shortcut-free metrics.
  """

  def get_potentially_guess(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return (
          row["r2(e2.null).blank.correct"]
          | row["r2(r1(e1.null)).blank.correct"]
      ) & row["r2(r1(e1)).blank.real.correct"]
    else:
      return (
          row["r2(e2.null).correct"] | row["r2(r1(e1.null)).correct"]
      ) & row["r2(r1(e1)).real.correct"]

  def get_both_correct(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return row["r1(e1).blank.correct"] & row["r2(e2).blank.correct"]
    else:
      return row["r1(e1).correct"] & row["r2(e2).correct"]

  def get_composability_denom(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return (
          row["both_correct"]
          & ~row["potentially_guess"]
          & (row["r2(r1(e1)).blank.unusable"] == False)  # pylint: disable=singleton-comparison
      )
    else:
      return (
          row["both_correct"]
          & ~row["potentially_guess"]
          & (row["r2(r1(e1)).unusable"] == False)  # pylint: disable=singleton-comparison
      )

  def get_composability_numer(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return row["r2(r1(e1)).blank.real.correct"] & row["composability_denom"]
    else:
      return row["r2(r1(e1)).real.correct"] & row["composability_denom"]

  df.loc[:, "potentially_guess"] = df.apply(
      lambda row: get_potentially_guess(row, blank_cols_already_processed),
      axis=1,
  )
  df.loc[:, "both_correct"] = df.apply(
      lambda row: get_both_correct(row, blank_cols_already_processed), axis=1
  )
  df.loc[:, "composability_denom"] = df.apply(
      lambda row: get_composability_denom(row, blank_cols_already_processed),
      axis=1,
  )
  df.loc[:, "composability_numer"] = df.apply(
      lambda row: get_composability_numer(row, blank_cols_already_processed),
      axis=1,
  )

  df.loc[:, "potentially_guess"] = df["potentially_guess"].astype(bool)
  df.loc[:, "both_correct"] = df["both_correct"].astype(bool)
  df.loc[:, "composability_denom"] = df["composability_denom"].astype(bool)
  df.loc[:, "composability_numer"] = df["composability_numer"].astype(bool)

  return df


def run_completion(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model_name_or_path: str,
    batch_size: int = 4,
    backend: str = "vllm",
    force_completion: bool = False,
) -> None:
  """Run model completion on the DataFrame.

  Args:
      df: The input DataFrame.
      model: The model to use.
      tokenizer: The tokenizer to use.
      model_name_or_path: The name or path of the model.
      batch_size: The batch size for processing.
      backend: The backend to use ("vllm" or "hf").
      force_completion: Whether to force completion even if it already exists.
  """
  if is_instruction_tuned(model, model_name_or_path):
    instruction_tuned = True
    print("Instruction-tuned model")
    fact_types = it_fact_types
  else:
    instruction_tuned = False
    print("Pretrained model")
    fact_types = pretrained_fact_types

  df.loc[:, "model"] = model_name_or_path
  df.loc[:, "model_type"] = (
      "instruction-tuned" if instruction_tuned else "pretrained"
  )

  for _, fact_type in fact_types.items():
    print(f"Running completion for {fact_type}")

    if backend == "hf":
      df = run_hf_completion(
          model,
          tokenizer,
          df,
          fact_type,
          instruction_tuned=instruction_tuned,
          batch_size=batch_size,
          force_completion=force_completion,
      )
    else:
      df = run_vllm_completion(
          model,
          df,
          fact_type,
          instruction_tuned=instruction_tuned,
          force_completion=force_completion,
      )
    gc.collect()


def evaluate_patchscopes(
    df: pd.DataFrame,
    fact_type: str,
    answer_entity_type: str,
    matches_col: str,
    correct_col: str,
    answer_postfix: str = "aliases",
    normalize: bool = True,
) -> pd.DataFrame:
  """Evaluate the patchscopes results for the given DataFrame and composition type.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
      answer_entity_type: The answer entity type.
      matches_col: The column name for matches.
      correct_col: The column name for correct answers.
      answer_postfix: The postfix for the answer column.
      normalize: Whether to normalize the text.

  Returns:
      The updated DataFrame with evaluation results.
  """
  df.loc[:, f"{fact_type}.completion"] = df.loc[
      :, f"{fact_type}.completion"
  ].apply(lambda x: "" if pd.isnull(x) else x)
  df.loc[:, f"{fact_type}.{matches_col}"] = df.apply(
      lambda row: get_matches(
          row[f"{fact_type}.completion"],
          row[f"{answer_entity_type}.{answer_postfix}"],
          normalize=normalize,
      ),
      axis=1,
  )
  df.loc[:, f"{fact_type}.{correct_col}"] = df.loc[
      :, f"{fact_type}.{matches_col}"
  ].apply(lambda x: len(x) > 0)  # pylint: disable=g-explicit-length-test

  df[f"{fact_type}.{correct_col}"] = df.loc[
      :, f"{fact_type}.{correct_col}"
  ].astype(bool)
  return df


def run_patchscopes_evaluation(
    df: pd.DataFrame,
    fact_type: str,
    source_layer_idxs: list[int],
    target_layer_idxs: list[int],
    num_return_sequences: int,
) -> pd.DataFrame:
  """Run evaluation of the Patchscopes results for the given DataFrame.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
      source_layer_idxs: The list of source layer indices.
      target_layer_idxs: The list of target layer indices.
      num_return_sequences: The number of return sequences.

  Returns:
      The updated DataFrame with patchscopes evaluation results.
  """
  t1_completion_cols = [
      col for col in df if "t1" in col and col.endswith(".completion")
  ]
  t2_completion_cols = [
      col for col in df if "t2" in col and col.endswith(".completion")
  ]
  for col in tqdm.tqdm(t1_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e2", "e2.matches", "e2.correct"
    )
  for col in tqdm.tqdm(t1_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e3", "e3.matches", "e3.correct"
    )
  for col in tqdm.tqdm(t2_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e2", "e2.matches", "e2.correct"
    )
  for col in tqdm.tqdm(t2_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e3", "e3.matches", "e3.correct"
    )

  def get_patchscopes_correct(row, fact_type, t, e, i, j):
    return any(
        row[f"{fact_type}.{t}-{k}-{i}-{j}.{e}.correct"]
        for k in range(num_return_sequences)
    )

  for t in ["t1", "t2"]:
    for e in ["e2", "e3"]:
      for i in source_layer_idxs:
        for j in target_layer_idxs:
          fn = functools.partial(
              get_patchscopes_correct,
              fact_type=fact_type,
              t=t,
              e=e,
              i=i,
              j=j,
          )
          df.loc[:, f"{fact_type}.{t}-{i}-{j}.{e}.correct"] = df.apply(
              fn, axis=1
          )

  return df

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

"""Utility functions for tokenization."""

import functools

import torch
import transformers


def to_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    target: str | list[str],
    prepend_bos: bool = True,
    padding_side: str = "left",
    return_tensors: str = "pt",
    return_original: bool = False,
) -> dict[str, torch.Tensor] | torch.Tensor:
  """Convert target text to tokens using the tokenizer.

  Args:
      tokenizer: The tokenizer to use.
      target: The target text to tokenize.
      prepend_bos: Whether to prepend the beginning-of-sequence token.
      padding_side: The side to pad the tokens.
      return_tensors: The format to return the tensors.
      return_original: Whether to return the original token dictionary.

  Returns:
      The tokenized target text.
  """
  orig_padding_side = tokenizer.padding_side
  tokenizer.padding_side = padding_side

  padding = True if return_tensors else False

  target_tokens = tokenizer(
      target,
      add_special_tokens=prepend_bos,
      return_tensors=return_tensors,
      padding=padding,
  )

  tokenizer.padding_side = orig_padding_side

  if return_original:
    return target_tokens

  return target_tokens["input_ids"]


def to_str_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    target: str | list[str],
    prepend_bos: bool = True,
) -> list[str] | list[list[str]]:
  """Convert target text to string tokens using the tokenizer.

  Args:
      tokenizer: The tokenizer to use.
      target: The target text to tokenize.
      prepend_bos: Whether to prepend the beginning-of-sequence token.

  Returns:
      The tokenized target text as strings.
  """
  tokens = to_tokens(tokenizer, target, prepend_bos, return_tensors=None)
  if isinstance(target, str):
    return [tokenizer.decode(t) for t in tokens]
  else:
    tokens_list = tokens
    return [[tokenizer.decode(t) for t in tokens] for tokens in tokens_list]


def to_first_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    target: str | list[str],
) -> torch.Tensor:
  """Convert target text to the first token using the tokenizer.

  Args:
      tokenizer: The tokenizer to use.
      target: The target text to tokenize.

  Returns:
      The first token of the tokenized target text.
  """
  tokens = to_tokens(tokenizer, target, prepend_bos=False, padding_side="right")
  return tokens[:, 0]


def to_first_str_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    target: str | list[str],
) -> str | list[str]:
  """Convert target text to the first string token using the tokenizer.

  Args:
      tokenizer: The tokenizer to use.
      target: The target text to tokenize.

  Returns:
      The first string token of the tokenized target text.
  """
  str_tokens = to_str_tokens(tokenizer, target, prepend_bos=False)
  if isinstance(target, str):
    return str_tokens[0]
  else:
    return [tokens[0] for tokens in str_tokens]


def requires_prepending_space(
    tokenizer: transformers.PreTrainedTokenizerBase,
    target: str,
) -> bool:
  """Check if the tokenizer requires prepending a space to the target text.

  Args:
      tokenizer: The tokenizer to use.
      target: The target text to check.

  Returns:
      True if the tokenizer requires prepending a space, False otherwise.
  """
  sacrificial_text = "to"
  preceding_char = " "

  encode = functools.partial(tokenizer.encode, add_special_tokens=False)
  assert len(encode("")) == 0  # pylint: disable=g-explicit-length-test
  bos_token_id = tokenizer.bos_token_id

  sacrificial_tokens = encode(sacrificial_text)
  n_sacrificial_tokens = len(sacrificial_tokens)
  preceding_token_id = encode(sacrificial_text + preceding_char)
  assert len(preceding_token_id) == 1 + n_sacrificial_tokens
  preceding_token_id = preceding_token_id[-1]
  target_tokens_with_sacrifice_and_space = encode(
      sacrificial_text + preceding_char + target
  )
  assert bos_token_id not in target_tokens_with_sacrifice_and_space
  assert (
      target_tokens_with_sacrifice_and_space[:n_sacrificial_tokens]
      == sacrificial_tokens
  )
  target_tokens_with_space = target_tokens_with_sacrifice_and_space[
      n_sacrificial_tokens:
  ]
  if target_tokens_with_space[0] == preceding_token_id:
    first_token_idx = 1
    first_token = target_tokens_with_space[first_token_idx]
    assert preceding_char not in tokenizer.decode([first_token])
    assert preceding_char not in tokenizer.decode([first_token, first_token])
    token_includes_preceding_char = False
  else:
    first_token_idx = 0
    first_token = target_tokens_with_space[first_token_idx]
    assert preceding_char in tokenizer.decode([first_token, first_token])
    token_includes_preceding_char = True
  return token_includes_preceding_char


def find_exact_substrings_token_positions_from_tensor(
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_token_ids: torch.Tensor,
    batch_substrings: str | list[str],
    only_last: bool = True,
) -> list[list[int]] | list[int]:
  """Find the positions of exact substrings in tokenized tensors.

  Args:
      tokenizer: The tokenizer to use.
      batch_token_ids: The batch of token IDs.
      batch_substrings: The batch of substrings to find.
      only_last: Whether to return only the last position.

  Returns:
      A list of lists containing the positions of the substrings when only_last
      is False,
      otherwise a list of integers containing the positions of the substrings.
  """
  batch_results = []
  assert len(batch_token_ids) == len(batch_substrings)
  if isinstance(batch_substrings[0], str):
    batch_substrings = [[substrings] for substrings in batch_substrings]
    single_substring = True
  else:
    single_substring = False

  for token_ids, substrings in zip(batch_token_ids, batch_substrings):

    string = tokenizer.decode(token_ids)
    string = string.replace(" ", "")

    results = []

    for substring in substrings:
      substring = substring.replace(" ", "")
      start_idx = string.find(substring)

      if start_idx == -1:  # If the substring is not found
        continue

      token_positions = []

      for i in range(len(token_ids)):
        partial_string = tokenizer.decode(token_ids[: i + 1]).replace(" ", "")

        if len(partial_string) > start_idx:
          token_positions.append(i)

        search_result = partial_string.find(substring)
        if search_result != -1:
          break

      if only_last:
        token_positions = token_positions[-1]
      results.append(token_positions)

    if single_substring:
      results = results[0]
    batch_results.append(results)

  return batch_results


def find_exact_substrings_token_positions_from_string(
    tokenizer: transformers.PreTrainedTokenizerBase,
    strings: str | list[str],
    substrings: str | list[str],
    only_last: bool = True,
    prepend_bos: bool = True,
    padding_side: str = "left",
) -> list[list[int]] | list[int]:
  """Find the positions of exact substrings in tokenized strings.

  Args:
      tokenizer: The tokenizer to use.
      strings: The strings to tokenize.
      substrings: The substrings to find.
      only_last: Whether to return only the last position.
      prepend_bos: Whether to prepend the beginning-of-sequence token.
      padding_side: The side to pad the tokens.

  Returns:
      A list of lists containing the positions of the substring tokens when
      only_last is False,
      otherwise a list of integers containing the positions of the substring
      tokens.
  """
  token_ids = to_tokens(
      tokenizer, strings, prepend_bos=prepend_bos, padding_side=padding_side
  )
  return find_exact_substrings_token_positions_from_tensor(
      tokenizer, token_ids, substrings, only_last
  )


def get_completion(
    generated: torch.Tensor,
    prompt_inputs: dict[str, torch.Tensor],
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> list[str]:
  """Get the completion text from the generated tokens.

  Args:
      generated: The generated tokens.
      prompt_inputs: The prompt inputs.
      tokenizer: The tokenizer to use.

  Returns:
      The completion text.
  """
  return tokenizer.batch_decode(
      generated[:, prompt_inputs["input_ids"].shape[1] :],
      skip_special_tokens=True,
  )


def get_subject_prompt(prompt: str, subject: str) -> str:
  """Get the subject prompt from the prompt text.

  Args:
      prompt: The prompt text.
      subject: The subject of the prompt to find.

  Returns:
      The prefix of the prompt cut at the end of the subject part.
  """
  start_index = prompt.lower().rindex(subject.lower())
  end_index = start_index + len(subject)
  return prompt[:end_index]

"""This module contains utility functions for the Patchscopes experiments.

Used in the paper "Do Large Language Models Perform Latent Multi-Hop Reasoning
without Exploiting Shortcuts?".
"""

from baukit import Trace
from baukit import TraceDict
from src import model_utils
from src import tokenization_utils
import torch
import tqdm
import transformers


def get_hidden_states(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
  """Get hidden states from the model for the given prompt inputs.

  Args:
      model: The model to get hidden states from.
      prompt_inputs: The prompt inputs.

  Returns:
      The hidden states as a torch.Tensor.
  """
  layers = model_utils.get_layer_names(model)
  with torch.no_grad(), TraceDict(model, layers) as trace:
    model(**prompt_inputs)
    hidden_states = torch.stack(
        [trace[layer].output[0].cpu() for layer in layers]
    )
  return hidden_states


@torch.no_grad()
def generate_with_patching_layer(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    target_prompt_inputs: dict[str, torch.Tensor],
    hidden_state: torch.Tensor,
    target_layer: str,
    target_position: int,
    do_sample: bool = False,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    max_new_tokens: int = 12,
) -> list[str]:
  """Generate completions with patched hidden states at a specific layer.

  Args:
      model: The model to use for generation.
      tokenizer: The tokenizer to use.
      target_prompt_inputs: The target prompt inputs.
      hidden_state: The hidden state to patch.
      target_layer: The target layer to patch.
      target_position: The target position to patch.
      do_sample: Whether to sample during generation.
      num_beams: The number of beams for beam search.
      num_return_sequences: The number of return sequences.
      max_new_tokens: The maximum number of new tokens to generate.

  Returns:
      The generated completions as a list of strings.
  """
  if max(num_beams, num_return_sequences) > 1:
    hidden_state = hidden_state.repeat_interleave(
        max(num_beams, num_return_sequences), dim=0
    )

  def replace_hidden_state_hook(output):
    hs = output[0]  # [B, L, D]
    if hs.shape[1] == 1:  # After first replacement the hidden state is cached
      return output

    hs[:, target_position, :] = hidden_state.to(hs.device)
    return (hs,) + output[1:]

  with torch.no_grad(), Trace(
      model,
      layer=target_layer,
      retain_output=False,
      edit_output=replace_hidden_state_hook,
  ):
    generated = model.generate(
        **target_prompt_inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )

  completions = tokenization_utils.get_completion(
      generated, target_prompt_inputs, tokenizer
  )
  if num_return_sequences > 1:
    # group completions
    completions = [
        completions[i * num_return_sequences : (i + 1) * num_return_sequences]
        for i in range(len(target_prompt_inputs["input_ids"]))
    ]
  return completions


@torch.no_grad()
def get_completions_from_patching(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompts: list[str],
    subject_prompts: list[str],
    target_prompts: list[str],
    do_sample: bool = True,
    num_return_sequences: int = 3,
    source_layer_idxs: list[int] = None,
    target_layer_idxs: list[int] = None,
) -> dict[tuple[str, int, int, int], list[str]]:
  """Get completions from patching hidden states at specific layers.

  Args:
      model: The model to use for generation.
      tokenizer: The tokenizer to use.
      prompts: The list of prompts.
      subject_prompts: The list of subject prompts.
      target_prompts: The list of target prompts.
      do_sample: Whether to sample during generation.
      num_return_sequences: The number of return sequences.
      source_layer_idxs: The list of source layer indices.
      target_layer_idxs: The list of target layer indices.

  Returns:
      A dictionary mapping (token_position, k, source_index, target_index) to
      generated completions.
  """
  prompt_inputs = tokenizer(
      prompts, return_tensors="pt", padding=True, truncation=True
  ).to(model.device)
  last_subject_token_positions = (
      tokenization_utils.find_exact_substrings_token_positions_from_tensor(
          tokenizer, prompt_inputs["input_ids"], subject_prompts
      )
  )

  hidden_states = get_hidden_states(model, prompt_inputs)
  t1_hidden_states = torch.stack(
      [
          hidden_states[:, i, pos, :]
          for i, pos in enumerate(last_subject_token_positions)
      ],
      dim=1,
  )
  t2_hidden_states = hidden_states[:, :, -1, :]  # [L, B, D]

  target_prompt_inputs = tokenizer(
      target_prompts, return_tensors="pt", padding=True, truncation=True
  ).to(model.device)
  target_position = target_prompt_inputs["input_ids"].shape[1] - 1

  completions = dict()
  pbar = tqdm.tqdm(enumerate(t1_hidden_states), leave=False)
  for i, t1_hidden_state in pbar:
    if i not in source_layer_idxs:
      continue
    for j, target_layer in enumerate(model_utils.get_layer_names(model)):
      if j not in target_layer_idxs:
        continue
      pbar.set_description(f"Patching {i} -> {j} at t1")
      generations = generate_with_patching_layer(
          model,
          tokenizer,
          target_prompt_inputs,
          t1_hidden_state,
          target_layer,
          target_position,
          do_sample=do_sample,
          num_return_sequences=num_return_sequences,
      )
      if num_return_sequences == 1:
        completions[("t1", 0, i, j)] = generations
      else:
        generations = list(map(list, zip(*generations)))
        for k, completion in enumerate(generations):
          completions[("t1", k, i, j)] = completion
      model_utils.flush()

  pbar = tqdm.tqdm(enumerate(t2_hidden_states), leave=False)
  for i, t2_hidden_state in pbar:
    if i not in source_layer_idxs:
      continue
    for j, target_layer in enumerate(model_utils.get_layer_names(model)):
      if j not in target_layer_idxs:
        continue
      pbar.set_description(f"Patching {i} -> {j} at t2")
      generations = generate_with_patching_layer(
          model,
          tokenizer,
          target_prompt_inputs,
          t2_hidden_state,
          target_layer,
          target_position,
          do_sample=do_sample,
          num_return_sequences=num_return_sequences,
      )
      if num_return_sequences == 1:
        completions[("t2", 0, i, j)] = generations
      else:
        generations = list(map(list, zip(*generations)))
        for k, completion in enumerate(generations):
          completions[("t2", k, i, j)] = completion
      model_utils.flush()

  return completions

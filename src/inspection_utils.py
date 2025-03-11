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

"""This module contains utility functions for running experiments to answer the research questions in the paper "Do Large Language Models Latently Perform Multi-Hop Reasoning?"."""

import ast
import re
from typing import Callable, Optional

from baukit.nethook import TraceDict
from fancy_einsum import einsum
import pandas as pd
from src import data_utils
from src import model_utils
from src import tokenization_utils
from src.data_utils import batchify
import torch
import torch.nn.functional as F
import tqdm
import transformers


def run_completion_and_evaluation(
    df: pd.DataFrame,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int,
) -> None:
  """Run completion and evaluation for different fact types in the dataset.

  Args:
      df: Dataframe containing the data to evaluate.
      model: The language model to use for completion.
      tokenizer: Tokenizer corresponding to the model.
      batch_size: Batch size for processing.

  Returns:
      None - Results are written directly to the dataframe.
  """
  for fact_type in ["r1(e1)", "r2(e2)", "r2(r1(e1))"]:
    answers_col = "e2.aliases" if fact_type == "r1(e1)" else "e3.aliases"
    fill_completion_and_evaluation(
        df,
        fact_type,
        answers_col,
        model,
        tokenizer,
        batch_size=batch_size,
        do_efficient_batchify=(fact_type != "r2(r1(e1))"),
    )


@torch.no_grad()
def run_rq1(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    df: pd.DataFrame,
    batch_size: int,
    inner_batch_size: int,
    entity_col: str,
    skip_positive: bool = False,
) -> None:
  """Run the experiment to answer RQ1 of "Do Large Language Models Latently Perform Multi-Hop Reasoning?".

  How often does an LLM perform the first hop of reasoning while processing
  two-hop prompts?

  Args:
      model: The language model to evaluate.
      tokenizer: Tokenizer corresponding to the model.
      df: Dataframe containing the experimental data.
      batch_size: Size of outer batches for processing.
      inner_batch_size: Size of inner batches that retrieve the residual
        streams.
      entity_col: Column name containing the bridge entities.
      skip_positive: Whether to skip calculation of RQ1 for the positive
        examples because it can be calculated when run_rq2 is called.

  Returns:
      None - Results are written directly to the dataframe.
  """

  neg_types = ["r2(r1(e1'))", "r2(r1'(e1))"]
  for neg_type in neg_types:
    neg_prompt_col = f"{neg_type}.prompt"
    neg_subject_col = f"{neg_type}.subject_cut.prompt"

    fill_full_latent_info(
        df,
        neg_prompt_col,
        entity_col,
        neg_subject_col,
        model,
        tokenizer,
        batch_size=batch_size,
        inner_batch_size=inner_batch_size,
    )

  if not skip_positive:
    positive_col = "r2(r1(e1)).prompt"
    positive_subject_col = "r2(r1(e1)).subject_cut.prompt"
    fill_full_latent_info(
        df,
        positive_col,
        entity_col,
        positive_subject_col,
        model,
        tokenizer,
        batch_size=batch_size,
        inner_batch_size=inner_batch_size,
    )


def run_appositive(
    df: pd.DataFrame,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int,
    inner_batch_size: int,
    gather_device: str,
) -> None:
  """Run the experiment in Appendix C of "Do Large Language Models Latently Perform Multi-Hop Reasoning?".

  This validates the internal entity recall metric used in RQ1 as a reasonable
  proxy for measuring internal entity recall. We check how often the model is
  likely to generate the bridge entity as an appositive after a comma, e.g.,
  "The mother of the singer of Superstition, Stevie Wonder" increases when the
  internal recall of the bridge entity increases at the last token of the
  descriptive mention of the bridge entity, "the singer of Superstition".

  Args:
      df: Dataframe containing the experimental data.
      model: The language model to evaluate.
      tokenizer: Tokenizer corresponding to the model.
      batch_size: Size of outer batches for processing.
      inner_batch_size: Size of inner batches that retrieve the residual
        streams.
      gather_device: Device to gather results on ("cpu" or "cuda:N").

  Returns:
      None - Results are written directly to the dataframe.
  """
  run_rq2(
      df,
      model,
      tokenizer,
      batch_size,
      inner_batch_size,
      gather_device,
      fact_type="r2(r1(e1)).appositive",
      entity_col="e2.value",
      answer_col="e2.value",
  )


@torch.no_grad()
def run_cot(
    df: pd.DataFrame,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int,
) -> None:
  """Run the experiment in Appendix D of "Do Large Language Models Latently Perform Multi-Hop Reasoning?".

  This provides empirical evidence for the consistency score used in RQ2 being a
  reasonable approximation of the model's utilization of knowledge about the
  bridge entity's attribute.

  Args:
      df: Dataframe containing the experimental data.
      model: The language model to evaluate.
      tokenizer: Tokenizer corresponding to the model.
      batch_size: Batch size for processing.

  Returns:
      None - Results are written directly to the dataframe.
  """
  model.eval()

  cot_prompt_cols = [
      "cot.r1(e1).therefore.prompt",
      "cot.r2(e2).therefore.prompt",
      "cot.r1(e1).r2(e2).therefore.prompt",
      "cot.r1(e1).prompt",
      "cot.r2(e2).prompt",
      "cot.r1(e1).r2(e2).prompt",
  ]
  for prompt_col in cot_prompt_cols:
    fill_consistency(
        df,
        "r2(e2).prompt",
        prompt_col,
        model,
        tokenizer,
        batch_size,
        tqdm_desc=f"[consistency w/ {prompt_col}]",
    )


def run_rq2(
    df: pd.DataFrame,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int = 16,
    inner_batch_size: int = 8,
    gather_device: str = "cuda:0",
    fact_type: str = "r2(r1(e1))",
    entity_col: str = "e2.value",
    answer_col: str = "e3.value",
) -> None:
  """Run the experiment to answer RQ2 of "Do Large Language Models Latently Perform Multi-Hop Reasoning?".

  How often does an LLM perform the second hop of reasoning while processing
  two-hop prompts? Specifically, we check how often the consistency between
  the output probability distributions of the one-hop and two-hop prompt
  increases when the internal recall of the bridge entity increases at the last
  token of the descriptive mention of the bridge entity in the two-hop prompt.

  Args:
      df: Dataframe containing the experimental data.
      model: The language model to evaluate.
      tokenizer: Tokenizer for the model.
      batch_size: Size of outer batches.
      inner_batch_size: Size of inner batches that retrieve the residual
        streams.
      gather_device: Device to gather results on ("cpu" or "cuda:N").
      fact_type: Type of fact to evaluate.
      entity_col: Column name for the bridge entities.
      answer_col: Column name for the answer entities of the one-hop and two-hop
        prompts.

  Returns:
      None - Results are written directly to the dataframe.
  """
  model.train()

  n_digits = len(str(model.config.num_hidden_layers))

  if fact_type == "r2(r1(e1))":
    experiment_type = "consistency"
    subject_col = f"{fact_type}.subject_cut.prompt"
  elif fact_type == "r2(r1(e1)).appositive":
    experiment_type = "appositive"
    subject_col = "r2(r1(e1)).subject_cut.prompt"
  else:
    raise ValueError(fact_type, entity_col, answer_col)

  prompt_col = f"{fact_type}.prompt"

  all_idxs = df.index.tolist()
  if f"layer00.gradient_consistency({prompt_col},r2(e2).prompt)" in df:
    all_idxs = df[
        (df[f"layer00.gradient_consistency({prompt_col},r2(e2).prompt)"] == "")  # pylint: disable=g-explicit-bool-comparison
        | (
            df[
                f"layer00.gradient_consistency({prompt_col},r2(e2).prompt)"
            ].isnull()
        )
    ].index.tolist()

  pbar = tqdm(
      range(
          0,
          (len(all_idxs) + batch_size - 1) // batch_size * batch_size,
          batch_size,
      )
  )
  for batch_start in pbar:
    batch_end = min(batch_start + batch_size, len(all_idxs))
    idxs = all_idxs[batch_start:batch_end]
    subdf = df.loc[idxs]

    prompts = subdf[prompt_col].tolist()
    singlehop_prompts = subdf["r2(e2).prompt"].tolist()
    subject_prompts = subdf[subject_col].tolist()
    bridge_entities = subdf[entity_col].tolist()
    targets = subdf[answer_col].tolist()

    pbar.set_description("preparing patch")
    xnew, alpha = get_xnew_and_alpha(
        prompts,
        subject_prompts,
        bridge_entities,
        model,
        tokenizer,
        gather_device,
        inner_batch_size,
        df=df,
        idxs=idxs,
        prompt_col=prompt_col,
        entity_col=entity_col,
    )

    # RQ2
    pbar.set_description(f"calculating gradient for {experiment_type}")
    hook_fn = get_adding_hook(xnew)
    if experiment_type == "consistency":
      outs = get_consistency_after_patching(
          prompts, singlehop_prompts, targets, model, tokenizer, hook_fn
      )
    elif experiment_type == "appositive":
      outs = get_appositive_after_patching(
          prompts, targets, model, tokenizer, hook_fn
      )
    else:
      raise ValueError(experiment_type)

    for dtype, value in outs.items():
      if dtype == "consistency":
        postfix = f"({prompt_col},r2(e2).prompt)"
      else:
        postfix = f"({answer_col}|{prompt_col})"

      # original metric
      col = f"{dtype}{postfix}"
      df.loc[idxs, col] = value.detach().cpu().tolist()

      # dmetric / dalpha
      grads = torch.autograd.grad(
          outputs=value,
          inputs=alpha,
          grad_outputs=torch.ones_like(value),
          create_graph=True,
      )
      for layer_idx, grad in enumerate(grads):
        col = f"layer{layer_idx:0{n_digits}d}.gradient_{dtype}{postfix}"
        df.loc[idxs, col] = grad.detach().cpu().tolist()
      del grads
    model_utils.flush()


def fill_full_latent_info(
    df: pd.DataFrame,
    prompt_col: str,
    subject_prompt_col: str,
    entity_col: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int,
    inner_batch_size: int,
) -> None:
  """Calculate and store internal entity recall in the dataframe.

  Args:
      df: Dataframe containing the data.
      prompt_col: Column name for prompts.
      subject_prompt_col: Column name for subject prompts.
      entity_col: Column name for the bridge entities (target of measuring the
        internal entity recall score).
      model: The language model.
      tokenizer: Tokenizer for the model.
      batch_size: Size of outer batches.
      inner_batch_size: Size of inner batches that retrieve the residual
        streams.

  Returns:
      None - Results are written directly to the dataframe
  """
  padding_side = "left"
  pos_slice = -1

  input_dict = {
      "prompts": df[
          subject_prompt_col
      ].tolist(),  # use subject prompts and pos_slice -1
      "targets": df[entity_col].tolist(),
  }

  batchified_get_latent_info = batchify(
      get_full_latent_info,
      batch_size=batch_size,
      concat_dim=1,  # concat along batch dimension
      tqdm_desc=f"[latentlogprob({entity_col}|{prompt_col})]",
  )
  latent_info = batchified_get_latent_info(
      input_dict,
      model=model,
      tokenizer=tokenizer,
      inner_batch_size=inner_batch_size,
      padding_side=padding_side,
      pos_slice=pos_slice,
      gather_device="cpu",
  )
  n_layers = model.config.num_hidden_layers
  n_digits = len(str(n_layers))

  for dtype, value in latent_info.items():
    for layer_idx, v in enumerate(value):
      df.loc[
          :,
          f"layer{layer_idx:0{n_digits}d}.latent{dtype}({entity_col}|{prompt_col})",
      ] = v.numpy().tolist()


def get_full_latent_info(
    prompts: list[str],
    targets: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    pos_slice: int | torch.Tensor,
    padding_side: str = "left",
    subject_prompts: Optional[list[str]] = None,
    inner_batch_size: int = 8,
    gather_device: str = "cpu",
    return_hidden_states: bool = False,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
  """Get internal entity recall and hidden states for target tokens in the prompts.

  Args:
      prompts: List of input prompts.
      targets: List of target strings to analyze.
      model: The language model.
      tokenizer: Tokenizer for the model.
      pos_slice: Position to slice the hidden states at, which should be the
        last token of the descriptive mention of the bridge entity in the
        two-hop prompt.
      padding_side: Side to apply padding ("left" or "right").
      subject_prompts: Optional list of subject prompts. If pos_slice is None,
        this is used to find the positions in get_resids.
      inner_batch_size: Batch size for inner processing that retrieves the
        residual streams.
      gather_device: Device to gather results on ("cpu" or "cuda:N").
      return_hidden_states: Whether to return hidden states.

  Returns:
      Dictionary containing:
          - 'logit': Logits for target tokens.
          - 'logprob': Log probabilities for target tokens (internal entity
          recall score).
          - 'hidden_states': Hidden states (if return_hidden_states=True).
  """
  if padding_side == "right":
    assert not isinstance(pos_slice, int)

  n_prompts = len(prompts)

  first_target_tokens = tokenization_utils.to_first_tokens(
      tokenizer, targets
  ).cpu()

  prompt_indexer = torch.arange(n_prompts)

  hidden_states = get_resids(
      prompts,
      subject_prompts,
      model,
      tokenizer,
      inner_batch_size,
      pos_slice=pos_slice,
  )  # [n_layers, inner_batch_size, *, d_model]

  final_ln = model_utils.get_final_ln(model)
  normed_hidden_states = [final_ln(h) for h in hidden_states]
  resids = normed_hidden_states

  target_token_logits = []
  target_token_logprobs = []
  for resid in resids:
    layer_logits = einsum(
        "inner_batch_size d_model, d_model d_vocab -> inner_batch_size d_vocab",
        resid,
        model_utils.get_unembedding_matrix(model, resid.device),
    )  # [inner_batch_size d_vocab]
    target_token_layer_logits = layer_logits[
        prompt_indexer, first_target_tokens
    ]  # [inner_batch_size] or [inner_batch_size, n_target_tokens]

    layer_logprobs = F.log_softmax(
        layer_logits, dim=-1
    )  # [inner_batch_size d_vocab]
    target_token_layer_logprobs = layer_logprobs[
        prompt_indexer, first_target_tokens
    ]  # [inner_batch_size] or [inner_batch_size, n_target_tokens]

    target_token_logits.append(target_token_layer_logits.to(gather_device))
    target_token_logprobs.append(target_token_layer_logprobs.to(gather_device))

  info = {
      "logit": torch.stack(target_token_logits),
      "logprob": torch.stack(target_token_logprobs),
  }
  if return_hidden_states:
    info["hidden_states"] = (
        hidden_states  # can contain tensors on multiple devices
    )

  return info  # be careful with GPU memory leak


def get_resids(
    prompts: list[str],
    subject_prompts: Optional[list[str]],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    inner_batch_size: int = 8,
    pos_slice: Optional[int | torch.Tensor] = -1,
    padding_side: str = "left",
) -> list[torch.Tensor]:
  """Get residual states from model layers for given prompts.

  Args:
      prompts: List of input prompts.
      subject_prompts: Optional list of subject prompts for finding the
        positions of the residual streams to return when pos_slice is None.
      model: The language model.
      tokenizer: Tokenizer for the model.
      inner_batch_size: Batch size for inner processing.
      pos_slice: Position to retrieve the residual streams from.
      padding_side: Side to apply padding ("left" or "right").

  Returns:
      List of residual stream tensors for each layer.
  """

  def inner_get_resids(
      inner_prompts, inner_subject_prompts=None, inner_pos_slice=None
  ):
    if inner_pos_slice is None:
      inner_pos_slice = (
          tokenization_utils.find_exact_substrings_token_positions_from_string(
              tokenizer,
              inner_prompts,
              inner_subject_prompts,
              only_last=True,
              prepend_bos=True,
              padding_side=padding_side,
          )
      )  # [n_prompts]

    prompt_inputs = tokenization_utils.to_tokens(
        tokenizer,
        inner_prompts,
        padding_side=padding_side,
        return_original=True,
    )
    layers = model_utils.get_layer_names(model)
    with TraceDict(model, layers=layers, retain_output=True) as trace:
      model(**prompt_inputs)

      resids = [trace[layer].output[0] for layer in layers]  # L * [B, N, D]

    if isinstance(inner_pos_slice, int):
      resids = [resid[:, inner_pos_slice, :] for resid in resids]
    elif isinstance(inner_pos_slice, (torch.Tensor, list, tuple)):
      n_prompts = resids[0].shape[0]
      assert len(inner_pos_slice) == n_prompts  # one position per prompt
      resids = [
          resid[torch.arange(n_prompts), inner_pos_slice, :] for resid in resids
      ]

    return resids

  inputs = {"inner_prompts": prompts}
  if subject_prompts is not None:
    inputs["inner_subject_prompts"] = subject_prompts

  kwargs = {}
  if pos_slice is not None and not isinstance(pos_slice, int):
    inputs["inner_pos_slice"] = pos_slice
  else:
    kwargs["inner_pos_slice"] = pos_slice

  if isinstance(prompts, (list, tuple)) and len(prompts) > inner_batch_size:
    return batchify(
        inner_get_resids,
        batch_size=inner_batch_size,
        concat_dim=0,
        tqdm_desc=None,
    )(inputs, **kwargs)
  else:
    return inner_get_resids(**inputs, **kwargs)


def fill_completion_and_evaluation(
    df: pd.DataFrame,
    fact_type: str,
    answers_col: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int,
    do_efficient_batchify: bool = True,
) -> None:
  """Fill completion predictions and evaluation metrics for a specific fact type.

  Args:
      df: Dataframe containing the data.
      fact_type: Type of fact to evaluate.
      answers_col: Column containing ground truth answers.
      model: The language model.
      tokenizer: Tokenizer for the model.
      batch_size: Size of batches for processing.
      do_efficient_batchify: Whether to use efficient batching.

  Returns:
      None - Results are written directly to the dataframe
  """
  prompt_col = f"{fact_type}.prompt"
  if do_efficient_batchify:
    get_batchified_result = data_utils.efficient_batchify(
        df,
        {"prompts": prompt_col, "answers": answers_col},
        get_completion_and_evaluation_result,
        batch_size=batch_size,
        tqdm_desc=f"[{fact_type}] generation",
    )
    result_dict = get_batchified_result(model=model, tokenizer=tokenizer)
    for dtype, value in result_dict.items():
      idxs, value = zip(*sorted(value.items(), key=lambda x: x[0]))
      df.loc[idxs, f"{fact_type}.{dtype}"] = value
  else:
    get_batchified_result = batchify(
        get_completion_and_evaluation_result,
        batch_size=batch_size,
        concat_dim=0,
        tqdm_desc=f"[{fact_type}] generation",
    )
    result_dict = get_batchified_result(
        {
            "prompts": df[prompt_col].tolist(),
            "answers": df[answers_col].tolist(),
        },
        model=model,
        tokenizer=tokenizer,
    )
    for dtype, value in result_dict.items():
      df.loc[:, f"{fact_type}.{dtype}"] = value

  df.astype({f"{fact_type}.correct": bool})
  df.loc[:, f"{fact_type}.matches"] = df.loc[:, f"{fact_type}.matches"].apply(
      ast.literal_eval
  )


@torch.no_grad()
def get_completion_and_evaluation_result(
    prompts: list[str],
    answers: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    max_new_tokens: int = 32,
) -> dict[str, list[str] | list[bool]]:
  """Generate completions and evaluate them against ground truth answers.

  Args:
      prompts: List of input prompts.
      answers: List of ground truth answers.
      model: The language model.
      tokenizer: Tokenizer for the model.
      max_new_tokens: Maximum number of tokens to generate.

  Returns:
      Dictionary containing:
      - 'completion': Generated completions.
      - 'matches': List of answers that appleared in the completions.
      - 'correct': Boolean indicating if any answer matched.
  """
  prompt_inputs = tokenization_utils.to_tokens(
      tokenizer, prompts, padding_side="left", return_original=True
  )

  generated = model.generate(
      **prompt_inputs,
      pad_token_id=tokenizer.eos_token_id,
      max_new_tokens=max_new_tokens,
      do_sample=False,
  )
  completions = tokenization_utils.get_completion(
      generated, prompt_inputs, tokenizer
  )

  matches = [
      get_simple_matches(completion, answer)
      for completion, answer in zip(completions, answers)
  ]
  correct = [len(matched_answer) > 0 for matched_answer in matches]  # pylint: disable=g-explicit-length-test

  matches = [str(a) for a in matches]

  return {
      "completion": completions,
      "matches": matches,
      "correct": correct,
  }


def get_appositive_after_patching(
    prompts: list[str],
    targets: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    hook_fn: Callable[[list[torch.Tensor], str], list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
  """Calculate logits and probabilities for appositive construction targets after applying patching.

  Args:
      prompts: List of input prompts (subject prompts followed by comma).
      targets: List of target strings (bridge entities).
      model: The language model.
      tokenizer: Tokenizer for the model.
      hook_fn: Function to apply patching to model activations.

  Returns:
      Dictionary containing:
          - 'logprob': Log probabilities for the first tokens of the bridge
          entities.
          - 'logit': Raw logits for the first tokens of the bridge entities.
  """
  first_target_tokens = tokenization_utils.to_first_tokens(
      tokenizer, targets
  ).cpu()
  prompt_indexer = torch.arange(len(prompts))

  prompt_inputs = tokenization_utils.to_tokens(
      tokenizer, prompts, padding_side="left", return_original=True
  )

  layers = model_utils.get_layer_names(model)
  with TraceDict(
      model, layers=layers, retain_output=False, edit_output=hook_fn
  ):
    outputs = model(**prompt_inputs)
    logits = outputs.logits[:, -1, :]

  logp = logits.log_softmax(-1)

  return {
      "logprob": logp[prompt_indexer, first_target_tokens],
      "logit": logits[prompt_indexer, first_target_tokens],
  }


def get_consistency_after_patching(
    multihop_prompts: list[str],
    singlehop_prompts: list[str],
    targets: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    hook_fn: Callable[[list[torch.Tensor], str], list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
  """Calculate consistency between multi-hop and single-hop predictions after applying patching.

  Args:
      multihop_prompts: List of multi-hop reasoning prompts.
      singlehop_prompts: List of single-hop reasoning prompts.
      targets: List of target strings (ground truth answers).
      model: The language model.
      tokenizer: Tokenizer for the model.
      hook_fn: Function to apply patching to model activations.

  Returns:
      Dictionary containing:
          - 'logprob': Log probabilities for target tokens computed for the
          multi-hop prompts.
          - 'consistency': Consistency scores between the output probabilities
          of the multi-hop and single-hop prompts.
  """
  first_target_tokens = tokenization_utils.to_first_tokens(
      tokenizer, targets
  ).cpu()
  prompt_indexer = torch.arange(len(multihop_prompts))

  multihop_prompt_inputs = tokenization_utils.to_tokens(
      tokenizer, multihop_prompts, padding_side="left", return_original=True
  )
  singlehop_prompt_inputs = tokenization_utils.to_tokens(
      tokenizer, singlehop_prompts, padding_side="left", return_original=True
  )

  layers = model_utils.get_layer_names(model)
  with TraceDict(
      model, layers=layers, retain_output=False, edit_output=hook_fn
  ):
    outputs = model(**multihop_prompt_inputs)
    multihop_logits = outputs.logits[:, -1, :]
  singlehop_logits = model(**singlehop_prompt_inputs).logits[:, -1, :]

  multihop_logp = multihop_logits.log_softmax(-1)
  singlehop_logp = singlehop_logits.log_softmax(-1)

  multihop_p = multihop_logp.exp()
  singlehop_p = singlehop_logp.exp()

  consistency = -0.5 * (
      -(multihop_p * singlehop_logp).sum(-1)
      + -(singlehop_p * multihop_logp).sum(-1)
  )
  return {
      "logprob": multihop_logp[prompt_indexer, first_target_tokens],
      "consistency": consistency,
  }


@torch.no_grad()
def fill_consistency(
    df: pd.DataFrame,
    singlehop_prompt_col: str,
    multihop_prompt_col: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int = 128,
    tqdm_desc: str = "consistency",
) -> None:
  """Calculate and store consistency scores between the output probabilities of the multi-hop and single-hop prompts.

  Args:
      df: Dataframe containing the data.
      singlehop_prompt_col: Column name for single-hop prompts.
      multihop_prompt_col: Column name for multi-hop prompts.
      model: The language model.
      tokenizer: Tokenizer for the model.
      batch_size: Size of batches for processing.
      tqdm_desc: Description for progress bar.

  Returns:
      None - Results are written directly to the dataframe.
  """
  batchified_get_consistency = batchify(
      get_consistency, batch_size=batch_size, concat_dim=0, tqdm_desc=tqdm_desc
  )
  consistency = batchified_get_consistency(
      {
          "prompts1": df[singlehop_prompt_col].tolist(),
          "prompts2": df[multihop_prompt_col].tolist(),
      },
      model=model,
      tokenizer=tokenizer,
  )

  df.loc[:, f"consistency({multihop_prompt_col},{singlehop_prompt_col})"] = (
      consistency.numpy()
  )


def get_consistency(
    prompts1: list[str],
    prompts2: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
) -> torch.Tensor:
  """Calculate the consistency scores between the output probabilities of the multi-hop and single-hop prompts.

  Computed as the average of the negative cross entropy between the predictions,
  which is a symmetric measure of the similarity between the two distributions.
  Please refer to Section 6.1 of "Do Large Language Models Latently Perform
  Multi-Hop Reasoning?" for the reasoning behind this design choice of the
  metric.

  Args:
      prompts1: First set of prompts.
      prompts2: Second set of prompts.
      model: The language model.
      tokenizer: Tokenizer for the model.

  Returns:
      Tensor of consistency scores.
  """
  return (
      -0.5
      * (
          get_ce(prompts1, prompts2, model, tokenizer)
          + get_ce(prompts2, prompts1, model, tokenizer)
      ).cpu()
  )


def get_ce(
    target_prompts: list[str],
    pred_prompts: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
) -> torch.Tensor:
  """Calculate cross entropy between the output probabilities of the target and prediction prompts.

  Args:
      target_prompts: Target prompts for calculating CE.
      pred_prompts: Prediction prompts for calculating CE.
      model: The language model.
      tokenizer: Tokenizer for the model.

  Returns:
      Cross entropy scores.
  """
  target_prompt_inputs = tokenization_utils.to_tokens(
      tokenizer, target_prompts, padding_side="left", return_original=True
  )
  pred_prompt_inputs = tokenization_utils.to_tokens(
      tokenizer, pred_prompts, padding_side="left", return_original=True
  )

  target_logits = model(**target_prompt_inputs).logits[:, -1, :]
  pred_logits = model(**pred_prompt_inputs).logits[:, -1, :]

  return -(target_logits.softmax(-1) * pred_logits.log_softmax(-1)).sum(-1)


def get_xnew_and_alpha(
    prompts: list[str],
    subject_prompts: list[str],
    bridge_entities: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    gather_device: str = "cuda:0",
    inner_batch_size: int = 8,
    df: Optional[pd.DataFrame] = None,
    idxs: Optional[list[int]] = None,
    prompt_col: Optional[str] = None,
    entity_col: Optional[str] = None,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
  """Calculate the patching directions and zero scaling factors for RQ2.

  Calculate patching directions that increase the internal recall of the bridge
  entities and dummy scaling factors (zero) to compute the gradient of
  consistency at the scaling factor of zero for efficient analysis of the model
  behavior. Please refer to Section 6.2 of "Do Large Language Models Latently
  Perform Multi-Hop Reasoning?" for more details.

  Args:
      prompts: List of input prompts.
      subject_prompts: List of subject prompts.
      bridge_entities: List of bridge entities.
      model: The language model.
      tokenizer: Tokenizer for the model.
      gather_device: Device to gather results on ("cpu" or "cuda:N").
      inner_batch_size: Batch size for inner processing that retrieves the
        residual streams.
      df: Optional dataframe to store results.
      idxs: Optional list of indices for dataframe.
      prompt_col: Optional column name for prompts. When provided, calculation
        of the positive examples of RQ1 is performed and stored in the
        dataframe.
      entity_col: Optional column name for the bridge entities. When provided,
        calculation of the positive examples of RQ1 is performed and stored in
        the dataframe.

  Returns:
      Tuple containing:
      - xnew: List of patching directions for each layer and prompt that
      increase the internal recall of the bridge entities.
      - alpha: List of scaling factors for each layer, which is set to zero.
  """
  pos_slice = (
      tokenization_utils.find_exact_substrings_token_positions_from_string(
          tokenizer,
          prompts,
          subject_prompts,
          only_last=True,
          prepend_bos=True,
          padding_side="left",
      )
  )  # [n_prompts]

  d_model = model.config.hidden_size
  n_prompts, n_pos = tokenization_utils.to_tokens(
      tokenizer, prompts, padding_side="left"
  ).shape

  calculated_g = get_calculated_g(
      prompts,
      bridge_entities,
      subject_prompts,
      model,
      tokenizer,
      None,
      gather_device,
      inner_batch_size,
      df=df,
      idxs=idxs,
      prompt_col=prompt_col,
      entity_col=entity_col,
  )
  n_layers = model.config.num_hidden_layers

  xnew = []
  alpha = []  # [n_layers * [n_prompts]]

  for i in range(n_layers):
    prompt_indexer = torch.arange(n_prompts)

    g = torch.zeros(
        (n_prompts, n_pos, d_model),
        device=calculated_g[i].device,
        requires_grad=True,
    )
    g = g.clone()

    g[prompt_indexer, pos_slice] += calculated_g[i]
    a = torch.zeros(
        n_prompts, device=calculated_g[i].device, requires_grad=True
    )

    alpha.append(a)
    xnew.append(a[:, None, None] * g)

  xnew = [
      [layer[i] for layer in xnew] for i in range(n_prompts)
  ]  # [n_prompts * [n_layers * [n_pos, d_model]]]

  return xnew, alpha


def get_calculated_g(
    prompts: list[str],
    bridge_entities: list[str],
    subject_prompts: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    pos_slice: Optional[int | torch.Tensor],
    gather_device: str,
    inner_batch_size: int = 8,
    df: Optional[pd.DataFrame] = None,
    idxs: Optional[list[int]] = None,
    prompt_col: Optional[str] = None,
    entity_col: Optional[str] = None,
) -> list[torch.Tensor]:
  """Calculate gradients of the consistency score with respect to the scaling factor at zero.

  The sign of the gradients tells whether increasing the internal recall of the
  bridge entity at the last token of the descriptive mention of the bridge
  entity in the two-hop prompt increases (positive) or decreases (negative) the
  consistency score between the output probabilities of the one-hop and two-hop
  prompts.

  Args:
      prompts: List of input prompts.
      bridge_entities: List of bridge entities.
      subject_prompts: List of subject prompts.
      model: The language model.
      tokenizer: Tokenizer for the model.
      pos_slice: Position to slice the hidden states at.
      gather_device: Device to gather results on.
      inner_batch_size: Batch size for inner processing.
      df: Optional dataframe to store results.
      idxs: Optional list of indices for dataframe.
      prompt_col: Optional column name for prompts.
      entity_col: Optional column name for entities.

  Returns:
      List of gradients for each layer.
  """
  outputs = get_full_latent_info(
      prompts,
      bridge_entities,
      model,
      tokenizer,
      pos_slice,
      padding_side="left",
      subject_prompts=subject_prompts,
      inner_batch_size=inner_batch_size,
      gather_device=gather_device,
      return_hidden_states=True,
  )
  patching_g = []

  # RQ1 positive entrec
  for layer_idx, (entrec, hidden) in enumerate(
      zip(outputs["logprob"], outputs["hidden_states"])
  ):
    if df is not None:
      postfix = f"({entity_col}|{prompt_col})"
      col = f"layer{layer_idx:02d}.latentlogprob{postfix}"
      df.loc[idxs, col] = entrec.detach().cpu().tolist()
    # hidden = hidden.to(gather_device)
    patching_g.append(
        torch.autograd.grad(
            outputs=entrec,
            inputs=hidden,
            grad_outputs=torch.ones_like(entrec),
            create_graph=True,
        )[0]
    )

  return patching_g


@torch.no_grad()
def get_adding_hook(
    xnew: list[
        list[torch.Tensor]
    ],  # [n_prompts * [n_layers * [n_pos, d_model]]]
) -> Callable[[list[torch.Tensor], str], list[torch.Tensor]]:
  """Create a hook function for patching model activations.

  Args:
      xnew: List of patching directions for each prompt and layer that is added
        to the original model activations.

  Returns:
      Hook function that adds the patching directions to model activations.
  """
  def resid_patching_hook(
      activation: list[torch.Tensor], layer_name: str
  ) -> list[torch.Tensor]:
    """Hook function that patches model activations.

    Args:
        activation: List of activation tensors (outputs of the hooked layer).
        layer_name: Name of the current layer.

    Returns:
        Patched activation tensors.
    """

    groups = re.fullmatch(r"[^\d]*(\d+)[^\d]*", layer_name)
    assert groups is not None, layer_name
    assert len(groups) == 1, groups
    layer_idx = int(groups[0])

    patch = torch.stack(
        [x[layer_idx] for x in xnew], dim=0
    )  # [n_prompts, n_pos, d_model]

    if activation[0].shape[1] < patch.shape[1]:
      return activation  # new token generation phase

    activation[0][:, :, :] += patch.to(activation[0].device)

    return activation

  return resid_patching_hook


def get_simple_matches(
    completion: str, correct_answers: list[list[str]]
) -> tuple[str, ...]:
  """Get simple exact match (EM) matches between the completion and correct answers.

  Args:
      completion: The generated completion.
      correct_answers: The list of correct answers.

  Returns:
      A tuple of matched answers.
  """
  assert isinstance(correct_answers, (list, tuple))
  if correct_answers:
    assert isinstance(correct_answers[0], (list, tuple))

  completion = completion.strip()

  matched_answers = []
  for answers in correct_answers:
    for possible_answer in answers:
      if completion.startswith(possible_answer):
        matched_answers.append(possible_answer)
    if not matched_answers:
      return ()
  return tuple(matched_answers)

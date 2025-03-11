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

"""Data utility functions."""

import ast
import collections
import functools
import itertools
import sys
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from src.model_utils import flush
import torch
import tqdm


def convert_object(x: Any) -> Any:
  """Convert a string representation of a list or dict to an actual list or dict."""
  if pd.isnull(x):
    return np.nan
  if x in ['', 'nan']:
    return np.nan
  if isinstance(x, str):
    return ast.literal_eval(x)
  return x


def convert_correct(x: Any) -> bool | float:
  """Convert a string representation of a boolean to an actual boolean."""
  if pd.isnull(x):
    return np.nan
  if x in ['', 'nan']:
    return np.nan
  if isinstance(x, str):
    return ast.literal_eval(x)
  if isinstance(x, bool):
    return x
  raise ValueError(f'Unknown correct value: {x}')


def read_dataframe(
    path: str,
    eval_cols: list[str] = (
        '.aliases',
        '.list',
        '.entities',
        '.keywords',
        '.matches',
    ),
    keep_default_na: bool = False,
) -> pd.DataFrame:
  """Read a CSV file into a pandas DataFrame and convert specific columns.

  Args:
      path: The path to the CSV file.
      eval_cols: List of column suffixes to evaluate.
      keep_default_na: Whether to keep default NaN values.

  Returns:
      A pandas DataFrame with converted columns.
  """
  df = pd.read_csv(path, keep_default_na=keep_default_na)
  for col in df.columns:
    if any([col.endswith(ec) for ec in eval_cols]):
      df.loc[:, col] = df[col].apply(convert_object)

  for e in ['e1', 'e2', 'e3']:
    if f'{e}.value' in df:
      df.loc[:, f'{e}.value'] = df[f'{e}.value'].astype(str)

  count_cols = [
      col
      for col in df
      if any(
          subs in col.split('.')[-1]
          for subs in ['count', 'c4', 'dolma', 'oscar', 'openwebtext']
      )
  ]
  for col in count_cols:
    df.loc[:, col] = df[col].apply(
        lambda x: int(float(x))
        if (x not in ['', 'nan']) and (not pd.isnull(x))
        else np.nan
    )

  correct_cols = [col for col in df if 'correct' in col.split('.')[-1]]
  for col in correct_cols:
    df.loc[:, col] = df[col].apply(convert_correct)

  if 'tid' in df:
    df = df.sort_values(by='tid')
  if 'eid' in df:
    df = df.sort_values(by='eid').reset_index(drop=True)

  return df


def get_efficient_batchified_info(
    df: pd.DataFrame, param_to_column: dict[str, str]
) -> tuple[list[list[int]], dict[str, list[Any]]]:
  """Get efficiently batchified information for efficient processing.

  Args:
      df: The input DataFrame.
      param_to_column: A dictionary mapping parameter names to column names.

  Returns:
      A tuple containing batched indices and batched parameter values.
  """
  columns = list(param_to_column.values())
  params = list(param_to_column.keys())

  assert df.index.is_unique

  subdfs = []
  inputs = []
  for values, subdf in tqdm.tqdm(df.groupby(columns), desc='making batches'):
    inputs.append(values)
    subdfs.append(subdf)
  inputs = list(zip(*inputs))

  param_to_column_values = {k: v for k, v in zip(params, inputs)}

  # sort inputs and subdfs by max length of inputs
  lengths = [len(x) for x in param_to_column_values[params[0]]]
  sorted_indices = np.argsort(lengths, kind='stable')[::-1]

  batched_param_to_column_values = {
      param: [param_to_column_values[param][i] for i in sorted_indices]
      for param in params
  }
  batched_indices = [subdfs[i].index.tolist() for i in sorted_indices]

  return batched_indices, batched_param_to_column_values


@torch.no_grad()
def efficient_batchify(
    df: pd.DataFrame,
    param_to_column: dict[str, str],
    function: Callable[..., Any],
    batch_size: int = 4,
    max_size: Optional[int] = None,
    tqdm_desc: Optional[str] = '',
    flush_step: Optional[int] = None,
    concat_dim: Optional[int] = None,
) -> Callable[..., Any]:
  """Efficiently batchify a function for processing a DataFrame.

  Args:
      df: The input DataFrame.
      param_to_column: A dictionary mapping parameter names to column names.
      function: The function to batchify.
      batch_size: The batch size for processing.
      max_size: The maximum size for processing.
      tqdm_desc: The description for the tqdm progress bar.
      flush_step: The step interval for flushing.
      concat_dim: The dimension for concatenation.

  Returns:
      A batchified function.
  """
  indices, param_to_column_values = get_efficient_batchified_info(
      df, param_to_column
  )

  @functools.wraps(function)
  def batchified_function(**kwargs):
    suboutputs = batchify(
        function,
        batch_size=batch_size,
        max_size=max_size,
        tqdm_desc=tqdm_desc,
        flush_step=flush_step,
        concat_dim=concat_dim,
    )(param_to_column_values, **kwargs)

    return unrolled_outputs(suboutputs, indices, concat_dim)

  return batchified_function


def unrolled_outputs(
    suboutputs: Any, batched_indices: list[list[int]], concat_dim: Optional[int]
) -> Any:
  """Unroll batched outputs to match the original indices.

  Args:
      suboutputs: The batched outputs.
      batched_indices: The batched indices.
      concat_dim: The dimension for concatenation.

  Returns:
      The unrolled outputs.
  """
  outputs = dict()
  if isinstance(suboutputs, (list, tuple)):
    for suboutput, indices in zip(suboutputs, batched_indices):
      for index in indices:
        outputs[index] = suboutput
  elif isinstance(suboutputs, dict):
    for k, subout in suboutputs.items():
      outs = unrolled_outputs(subout, batched_indices, concat_dim)
      outputs[k] = outs
  elif isinstance(suboutputs, torch.Tensor):
    if concat_dim != 0:
      suboutputs = suboutputs.transpose(0, concat_dim)
    for suboutput, indices in zip(suboutputs, batched_indices):
      for index in indices:
        outputs[index] = suboutput
  else:
    raise NotImplementedError
  return outputs


@torch.no_grad()
def batchify(
    function: Callable[..., Any],
    batch_size: int = 4,
    max_size: Optional[int] = None,
    tqdm_desc: Optional[str] = '',
    concat_dim: Optional[int] = None,
    flush_step: Optional[int] = None,
) -> Callable[..., Any]:
  """Batchify a function for processing.

  Args:
      function: The function to batchify.
      batch_size: The batch size for processing.
      max_size: The maximum size for processing.
      tqdm_desc: The description for the tqdm progress bar.
      concat_dim: The dimension for torch tensor concatenation.
      flush_step: The step interval for flushing.

  Returns:
      A batchified function.
  """

  @functools.wraps(function)
  def batchified_function(
      inputs: dict[str, list[Any]] | list[Any], **kwargs
  ) -> Any:
    results = []

    if isinstance(inputs, dict):
      upper_bound = len(inputs[list(inputs.keys())[0]])
    else:
      upper_bound = len(inputs)

    if upper_bound % batch_size > 0:
      upper_bound = (upper_bound // batch_size + 1) * batch_size

    iter_step = 0
    for start in tqdm.tqdm(
        range(0, upper_bound, batch_size),
        file=sys.stdout,
        disable=True if tqdm_desc is None else False,
        desc=tqdm_desc,
    ):
      try:
        if max_size and start > max_size:
          break

        if start + batch_size > upper_bound:
          local_batch_size = upper_bound - start
        else:
          local_batch_size = batch_size

        if isinstance(inputs, dict):
          batched_inputs = {
              k: v[start : start + local_batch_size] for k, v in inputs.items()
          }
          results.append(function(**batched_inputs, **kwargs))
        else:
          batched_inputs = inputs[start : start + local_batch_size]
          results.append(function(batched_inputs, **kwargs))

        iter_step += 1
        if flush_step is not None and iter_step % flush_step == 0:
          flush()
      except KeyboardInterrupt:
        print('KeyboardInterrupt at iter_step', iter_step)
        break

    return aggregated_results(results, concat_dim)

  return batchified_function


def aggregated_results(results: list[Any], concat_dim: Optional[int]) -> Any:
  """Aggregate batched results.

  Args:
      results: The batched results.
      concat_dim: The dimension for concatenation.

  Returns:
      The aggregated results.
  """
  # flatten the result
  if isinstance(results[0], torch.Tensor):
    results = torch.cat(results, dim=concat_dim)
  elif isinstance(results[0], (list, tuple)):
    if concat_dim is not None and isinstance(results[0][0], torch.Tensor):
      # assert isinstance(results[0][0], torch.Tensor)
      outputs = [[] for _ in range(len(results[0]))]
      for result in results:
        for i, r in enumerate(result):
          outputs[i].append(r)
      for i in range(len(outputs)):
        outputs[i] = torch.cat(outputs[i], dim=concat_dim)
      results = outputs
    else:
      results = list(itertools.chain.from_iterable(results))
  elif isinstance(results[0], dict):
    new_results = collections.defaultdict(list)
    keys = results[0].keys()
    for k in keys:
      v = [result[k] for result in results]
      v = aggregated_results(v, concat_dim)
      new_results[k] = v
    results = dict(new_results)
  else:
    raise NotImplementedError
  return results

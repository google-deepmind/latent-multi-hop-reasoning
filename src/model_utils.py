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

"""Utility functions for working with transformer models."""

import gc
import random

import numpy as np
import torch
import transformers


# Global cache for unembedding matrices on specific devices,
# e.g., _tensor_cache[("Qwen2ForCausalLM", "cuda:0")]
_tensor_cache: dict[tuple[str, str], torch.Tensor] = {}


def get_unembedding_matrix(
    model: transformers.PreTrainedModel, device: torch.device
) -> torch.Tensor:
  """Get or cache unembedding matrix for model on specified device.

  Args:
      model: The model to get the unembedding matrix from.
      device: The device to move the unembedding matrix to.

  Returns:
      The unembedding matrix as a torch.Tensor.
  """
  key = (type(model).__name__, str(device))

  if key not in _tensor_cache:
    # Get weight based on model type
    if isinstance(model, transformers.GPT2LMHeadModel):
      weight = model.transformer.wte.weight
    elif isinstance(model, transformers.LlamaForCausalLM):
      weight = model.lm_head.weight.T
    elif isinstance(model, transformers.Qwen2ForCausalLM):
      weight = model.lm_head.weight.T
    else:
      raise ValueError(
          f"Check the structure of {type(model)} and update this function"
          " accordingly to use the model"
      )

    _tensor_cache[key] = weight.to(device)

  return _tensor_cache[key]


def get_final_ln(model: transformers.PreTrainedModel) -> torch.nn.Module:
  """Get the final layer normalization module of the model.

  Args:
      model: The model to get the final layer normalization from.

  Returns:
      The final layer normalization module.
  """
  if isinstance(model, transformers.GPT2LMHeadModel):
    return model.ln_f

  if isinstance(
      model, (transformers.LlamaForCausalLM, transformers.Qwen2ForCausalLM)
  ):
    return model.model.norm

  raise ValueError(
      f"Check the structure of {type(model)} and update this function"
      " accordingly to use the model"
  )


def get_layers(model: transformers.PreTrainedModel) -> list[torch.nn.Module]:
  """Get the layers of the model.

  Args:
      model: The model to get the layers from.

  Returns:
      The layers of the model as a torch.nn.ModuleList.
  """
  if isinstance(model, transformers.GPT2LMHeadModel):
    return model.transformer.h

  if isinstance(
      model, (transformers.LlamaForCausalLM, transformers.Qwen2ForCausalLM)
  ):
    return model.model.transformer.layers

  raise ValueError(
      f"Check the structure of {type(model)} and update this function"
      " accordingly to use the model"
  )


def get_layer_names(model: transformers.PreTrainedModel) -> list[str]:
  """Get the names of the layers of the model.

  Args:
      model: The model to get the layer names from.

  Returns:
      A list of layer names.
  """
  if isinstance(model, transformers.GPT2LMHeadModel):
    return [f"transformer.h.{i}" for i in range(model.config.num_hidden_layers)]

  if isinstance(
      model, (transformers.LlamaForCausalLM, transformers.Qwen2ForCausalLM)
  ):
    return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]

  raise ValueError(
      f"Check the structure of {type(model)} and update this function"
      " accordingly to use the model"
  )


def is_instruction_tuned(
    model: transformers.PreTrainedModel, model_name_or_path: str
) -> bool:
  """Check if the model is instruction-tuned based on its name or path.

  Args:
      model: The model to check.
      model_name_or_path: The name or path of the model.

  Returns:
      True if the model is instruction-tuned, False otherwise.
  """
  if not isinstance(model, transformers.PreTrainedModel):  # vllm models
    print(
        "Make sure to check the naming pattern of the instruction-tuned model"
        f" of {model_name_or_path} in this function to ensure correct usage"
    )
  else:
    if not isinstance(
        model,
        (
            transformers.GPT2LMHeadModel,
            transformers.LlamaForCausalLM,
            transformers.Qwen2ForCausalLM,
            transformers.MistralForCausalLM,
            transformers.MixtralForCausalLM,
            transformers.OlmoForCausalLM,
            transformers.GemmaForCausalLM,
            transformers.Gemma2ForCausalLM,
            transformers.CohereForCausalLM,
        ),
    ):
      raise ValueError(
          "Check the naming pattern of the instruction-tuned model of"
          f" {model_name_or_path} and update this function accordingly to use"
          " the model"
      )

  return any([
      s in model_name_or_path
      for s in [
          "-Instruct",
          "-instruct",
          "-it",
          "-chat",
          "-Chat",
          "command-r",
          "gpt",
          "claude",
          "gemini",
      ]
  ])


def get_messages_start_role(model_name_or_path: str) -> str:
  """Get the starting role for messages of instruction-tuned models based on the model name.

  Args:
      model_name_or_path: The name or path of the model.

  Returns:
      The starting role for messages.
  """
  # e.g., "models/google/gemma-2b" -> "google/gemma-2b"
  model_name = "/".join(model_name_or_path.strip("/").split("/")[-2:])

  if model_name.startswith("google/gemma"):
    return "user"
  if model_name.startswith("mistralai/Mixtral"):
    return "user"
  if model_name.startswith("mistralai/Mistral"):
    return "system"
  if model_name.startswith("meta-llama/Meta-Llama") or model_name.startswith(
      "meta-llama/Llama"
  ):
    return "system"
  if model_name.startswith("Qwen/Qwen"):
    return "system"
  if model_name.startswith("01-ai/Yi"):
    return "system"
  if model_name.startswith("allenai/OLMo"):
    return "user"
  raise ValueError(
      f"Check the configuration of {model_name_or_path} and update this"
      " function accordingly to use the model"
  )


def flush() -> None:
  """Flush the GPU memory."""
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()


def set_random_seed(seed: int) -> None:
  """Set the random seed for reproducibility.

  Args:
      seed: The random seed to set.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

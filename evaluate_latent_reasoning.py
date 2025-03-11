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

"""Run experiments to perform shortcut-free evaluation of latent multi-hop reasoning as described in the paper 'Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?'."""

import argparse
import os
import traceback
from src import data_utils
from src import evaluation_utils
from src import model_utils
import transformers
from vllm import LLM

AutoModelForCausalLM = transformers.AutoModelForCausalLM
AutoTokenizer = transformers.AutoTokenizer


def get_parser():
  """Return an argument parser."""
  parser = argparse.ArgumentParser(
      description=(
          "Run experiments to perform shortcut-free evaluation of latent"
          " multi-hop reasoning as described in the paper 'Do Large Language"
          " Models Perform Latent Multi-Hop Reasoning without Exploiting"
          " Shortcuts?'"
      )
  )

  parser.add_argument(
      "--model_name_or_path",
      type=str,
      default="mistralai/Mistral-7B-v0.3",
      help=(
          "Path to pretrained model or model identifier from"
          " huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--revision",
      type=str,
      default=None,
      help="Specific model revision to use from HuggingFace.",
  )
  parser.add_argument(
      "--input_csv_path",
      type=str,
      default="datasets/SOCRATES.csv",
      help="Path to the input CSV file containing the evaluation dataset.",
  )
  parser.add_argument(
      "--hf_token",
      type=str,
      default=os.environ.get("HF_TOKEN", None),
      help=(
          "HuggingFace token for accessing models. Defaults to environment"
          " variable HF_TOKEN."
      ),
  )
  parser.add_argument(
      "--backend",
      type=str,
      default="vllm",
      choices=["vllm", "hf"],
      help=(
          "Backend to use for inference: 'vllm' for vLLM's optimized inference"
          " or 'hf' for inference using HuggingFace Transformers."
      ),
  )
  parser.add_argument(
      "--tensor_parallel_size",
      type=int,
      default=1,
      help=(
          "Number of GPUs to use for vLLM. Ignored for HuggingFace"
          " Transformers."
      ),
  )
  parser.add_argument(
      "--force_completion",
      action="store_true",
      help=(
          "Force the model to recompute completions even if they are already"
          " stored in the dataset."
      ),
  )
  parser.add_argument(
      "--gpu_memory_utilization",
      type=float,
      default=0.85,
      help=(
          "Target GPU memory utilization ratio for vLLM (between 0 and 1)."
          " Ignored for HuggingFace Transformers."
      ),
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=128,
      help=(
          "Number of queries to process in parallel when HuggingFace"
          " Transformers is used. Ignored for vLLM."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="results/evaluate_latent_reasoning",
      help="Directory to save evaluation results.",
  )
  return parser


def main(args):
  model_utils.set_random_seed(42)
  print(vars(args))

  model_name_or_path = args.model_name_or_path
  model_name = "/".join(model_name_or_path.strip("/").split("/")[-2:])
  safe_model_name = model_name.replace("/", "--")

  print(f"Loading {model_name_or_path}")
  kwargs = {}
  if args.revision:
    kwargs = {"revision": args.revision}
    print(f"Revision: {args.revision}")
    safe_model_name += f".{args.revision}"

  if args.backend == "hf":
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        attn_implementation="sdpa",
        token=args.hf_token,
        **kwargs,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=args.hf_token,
    )

    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
  elif args.backend == "vllm":
    if os.environ.get("HF_TOKEN", None) is None:
      os.environ["HF_TOKEN"] = args.hf_token

    model = LLM(
        model=f"{model_name_or_path}",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **kwargs,
    )
    tokenizer = model.get_tokenizer()
  else:
    raise ValueError(f"Unknown backend: {args.backend}")

  input_csv_path = args.input_csv_path
  safe_model_name_or_path = "/".join(
      model_name_or_path.strip("/").split("/")[-2:]
  )
  safe_model_name_or_path = safe_model_name_or_path.replace("/", "--")
  input_csv_name = os.path.basename(input_csv_path).replace(".csv", "")

  experiment_dir = args.output_dir
  os.makedirs(experiment_dir, exist_ok=True)
  print(f"Saving results to {experiment_dir}")

  output_csv_path = os.path.join(
      experiment_dir, f"{input_csv_name}.{safe_model_name_or_path}.csv"
  )

  print(f"Reading {input_csv_path}")
  df = data_utils.read_dataframe(input_csv_path)

  try:
    evaluation_utils.run_completion(
        df,
        model,
        tokenizer,
        model_name_or_path,
        batch_size=args.batch_size,
        backend=args.backend,
        force_completion=args.force_completion,
    )
    evaluation_utils.run_shortcut_free_evaluation(df)
    df = evaluation_utils.get_df_with_shortcut_free_metrics(df)

  except:  # pylint: disable=bare-except
    print(traceback.format_exc())
    output_csv_path = os.path.join(
        experiment_dir,
        f"{input_csv_name}.{safe_model_name_or_path}.completion.incomplete.csv",
    )

  df.to_csv(output_csv_path, index=False)
  print(f"Saved the dataframe to {output_csv_path}")


if __name__ == "__main__":
  main(get_parser().parse_args())

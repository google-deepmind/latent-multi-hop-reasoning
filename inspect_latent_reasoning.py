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

"""Runs experiments to inspect latent multi-hop reasoning in LLMs as described in the paper 'Do Large Language Models Latently Perform Multi-Hop Reasoning?'."""

import argparse
import datetime
import os
import traceback

from src import data_utils
from src import inspection_utils
from src import model_utils
from src import tokenization_utils
import transformers
import yaml

datetime = datetime.datetime
AutoModelForCausalLM = transformers.AutoModelForCausalLM
AutoTokenizer = transformers.AutoTokenizer


def get_parser():
  """Get the argument parser for the script."""
  parser = argparse.ArgumentParser(
      description=(
          "Run experiments to inspect latent multi-hop reasoning in "
          "large language models as described in the paper 'Do Large Language "
          "Models Latently Perform Multi-Hop Reasoning?'"
      )
  )

  parser.add_argument(
      "--model_name_or_path",
      type=str,
      default="mistralai/Mistral-7B-v0.3",
      help=(
          "Path to pretrained model or model identifier from "
          "huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--input_csv_path",
      type=str,
      default="datasets/TwoHopFact.csv",
      help="Path to the input CSV file containing the dataset.",
  )
  parser.add_argument(
      "--rq1_batch_size",
      type=int,
      default=512,
      help="Batch size for processing RQ1 experiments (first hop analysis).",
  )
  parser.add_argument(
      "--rq2_batch_size",
      type=int,
      default=32,
      help="Batch size for processing RQ2 experiments (second hop analysis).",
  )
  parser.add_argument(
      "--completion_batch_size",
      type=int,
      default=128,
      help="Batch size for generating completions.",
  )
  parser.add_argument(
      "--gather_device",
      type=str,
      default="cuda:0",
      help="Device to gather results on (e.g., 'cuda:0', 'cpu').",
  )
  parser.add_argument(
      "--hf_token",
      type=str,
      default=os.environ.get("HF_TOKEN_PATH", None),
      help=(
          "HuggingFace token for accessing models. Defaults to environment "
          "variable HF_TOKEN_PATH."
      ),
  )
  parser.add_argument(
      "--run_rq1",
      action="store_true",
      help="Run Research Question 1 experiments analyzing first hop reasoning.",
  )
  parser.add_argument(
      "--run_rq2",
      action="store_true",
      help=(
          "Run Research Question 2 experiments analyzing second hop reasoning."
      ),
  )
  parser.add_argument(
      "--run_appositive",
      action="store_true",
      help=(
          "Run appositive construction experiments for validating the "
          "internal entity recall metric."
      ),
  )
  parser.add_argument(
      "--run_cot",
      action="store_true",
      help="Run chain-of-thought experiments for validating consistency score.",
  )
  parser.add_argument(
      "--run_completion",
      action="store_true",
      help="Run completion generation and evaluation.",
  )
  parser.add_argument(
      "--half_precision",
      action="store_true",
      help="Use half precision for model weights.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="results/inspect_latent_reasoning",
      help="Directory to save experiment results.",
  )
  return parser


def main(args) -> None:
  model_utils.set_random_seed(42)
  print(vars(args))

  model_name_or_path = args.model_name_or_path
  input_csv_path = args.input_csv_path
  rq1_batch_size = args.rq1_batch_size
  rq2_batch_size = args.rq2_batch_size
  completion_batch_size = args.completion_batch_size
  gather_device = args.gather_device
  hf_token = args.hf_token
  half_precision = args.half_precision
  output_dir = args.output_dir

  model = AutoModelForCausalLM.from_pretrained(
      model_name_or_path,
      device_map="auto",
      token=hf_token,
  )
  if half_precision:
    model = model.half()

  tokenizer = AutoTokenizer.from_pretrained(
      model_name_or_path,
      token=hf_token,
  )
  tokenizer.padding_side = "left"
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

  # read input file
  safe_model_name_or_path = "/".join(
      model_name_or_path.strip("/").split("/")[-2:]
  )
  safe_model_name_or_path = safe_model_name_or_path.replace("/", "--")
  input_csv_name = os.path.basename(input_csv_path).replace(".csv", "")

  now = datetime.now()
  dt_string = now.strftime("%y%m%d_%H%M%S")
  experiment_dir = os.path.join(output_dir, dt_string)
  os.makedirs(experiment_dir, exist_ok=True)
  print(f"Saving results to {experiment_dir}")

  # Save args to a yaml file
  with open(os.path.join(experiment_dir, "args.yaml"), "w") as f:
    yaml.dump(vars(args), f)

  print(f"reading {input_csv_path}")
  df = data_utils.read_dataframe(input_csv_path)

  if tokenization_utils.requires_prepending_space(tokenizer, "Rome"):
    print("Prepending space to target")
    for col in df.columns:
      if col.endswith(".value"):
        df.loc[:, col] = df[col].apply(lambda x: f" {x}")
      if col.endswith(".aliases"):
        df.loc[:, col] = df[col].apply(
            lambda items: tuple(
                [tuple(f" {it}" for it in item) for item in items]
            )
        )

  # for analysis
  for col in df.columns:
    if col.endswith("value") and col + "_token" not in df:
      df.loc[:, col + "_token"] = df[col].apply(
          lambda x: tokenization_utils.to_first_str_tokens(tokenizer, x)
      )

  df = df.sort_values(
      by="r2(r1(e1)).prompt", key=lambda x: x.str.len(), ascending=False
  )

  if args.run_rq1:
    try:
      inspection_utils.run_rq1(
          model,
          tokenizer,
          df,
          rq1_batch_size,
          rq1_batch_size // 4,
          "e2.value",
          skip_positive=args.run_rq2,
      )
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.rq1.csv",
      )
    except:  # pylint: disable=bare-except
      print(traceback.format_exc())
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.rq1.incomplete.csv",
      )

    df.to_csv(output_csv_path, index=False)
    print(f"Saved {output_csv_path}")

  if args.run_rq2:
    try:
      inspection_utils.run_rq2(
          df,
          model,
          tokenizer,
          rq2_batch_size,
          rq2_batch_size // 4,
          gather_device,
          fact_type="r2(r1(e1))",
          entity_col="e2.value",
          answer_col="e3.value",
      )
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.rq2.csv",
      )
    except:  # pylint: disable=bare-except
      print(traceback.format_exc())
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.rq2.incomplete.csv",
      )

    df.to_csv(output_csv_path, index=False)
    print(f"Saved {output_csv_path}")

  if args.run_appositive:
    try:
      inspection_utils.run_appositive(
          df,
          model,
          tokenizer,
          rq2_batch_size,
          rq2_batch_size // 4,
          gather_device,
      )
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.appositive.csv",
      )
    except:  # pylint: disable=bare-except
      print(traceback.format_exc())
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.appositive.incomplete.csv",
      )

    df.to_csv(output_csv_path, index=False)
    print(f"Saved {output_csv_path}")

  if args.run_cot:
    try:
      inspection_utils.run_cot(df, model, tokenizer, completion_batch_size // 4)
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.cot.csv",
      )
    except:  # pylint: disable=bare-except
      print(traceback.format_exc())
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.cot.incomplete.csv",
      )

    df.to_csv(output_csv_path, index=False)
    print(f"Saved {output_csv_path}")

  if args.run_completion:
    try:
      inspection_utils.run_completion_and_evaluation(
          df, model, tokenizer, completion_batch_size
      )
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.completion.csv",
      )
    except:  # pylint: disable=bare-except
      print(traceback.format_exc())
      output_csv_path = os.path.join(
          experiment_dir,
          f"{input_csv_name}.{safe_model_name_or_path}.completion.incomplete.csv",
      )

    df.to_csv(output_csv_path, index=False)
    print(f"Saved {output_csv_path}")

  # save final output file
  output_csv_path = os.path.join(
      experiment_dir, f"{input_csv_name}.{safe_model_name_or_path}.csv"
  )
  df = df.sort_values(by="uid")
  df.to_csv(output_csv_path, index=False)
  print(f"Saved {output_csv_path}")


if __name__ == "__main__":
  main(get_parser().parse_args())

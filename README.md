# latent-multi-hop-reasoning

This repository contains the code and datasets used in the following two papers:

- Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva*, Sebastian Riedel*. [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837). In ACL 2024.
- Sohee Yang, Nora Kassner, Elena Gribovskaya, Sebastian Riedel*, Mor Geva*. [Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](https://arxiv.org/abs/2411.16679). arXiv, 2024.

## Installation
```bash
# Create and activate conda environment
conda create -n reasoning python=3.10
conda activate reasoning

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token environment variable (required for accessing certain models)
export HF_TOKEN="your_token_here"
```

## Datasets

The datasets are under the `datasets` directory.

### TwoHopFact

- Introduced in [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837)
- Contains 45,595 pairs of one-hop and two-hop factual prompts of 52 fact composition types with balanced distribution, designed to probe the internal mechanism of latent multi-hop reasoning
- `datasets/TwoHopFact.csv` (91MB)
- TwoHopFact is also available in huggingface datasets as [soheeyang/TwoHopFact](https://huggingface.co/datasets/soheeyang/TwoHopFact).

### SOCRATES (ShOrtCut-fRee lATent rEaSoning)

- Introduced in [Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](https://arxiv.org/abs/2411.16679)
- Contains 7,232 pairs of one-hop and two-hop factual prompts of 17 fact composition types, carefully created to evaluate latent multi-hop reasoning ability of LLMs with accuracy-based metrics while minimizing the risk of shortcuts
- `datasets/SOCRATES_v1.csv` (14MB): A cleaned-up version of the dataset which does not contain grammatical errors.
- `datasets/SOCRATES_v0.csv` (14MB): Used for the experiments in the paper which contains a few grammatical errors.
- SOCRATES v1 is also available in huggingface as [soheeyang/SOCRATES](https://huggingface.co/datasets/soheeyang/SOCRATES).

## Code Usage

### [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837)

**Inspection of Latent Multi-Hop Reasoning Pathway**

```bash
python inspect_latent_reasoning.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --input_csv_path datasets/TwoHopFact.csv \
    --rq1_batch_size 256 \
    --rq2_batch_size 8 \
    --completion_batch_size 64 \
    --hf_token $HF_TOKEN_PATH \
    --run_rq1 --run_rq2 --run_appositive --run_cot --run_completion
```

### [Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](https://arxiv.org/abs/2411.16679)

**Shortcut-Free Evaluation**

```bash
python evaluate_latent_reasoning.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --input_csv_path datasets/SOCRATES.csv \
    --tensor_parallel_size 2 \
    --batch_size 256 \
    --hf_token $HF_TOKEN_PATH
```

**Patchscopes Analysis**

```bash
python run_patchscopes.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --input_csv_path datasets/SOCRATES.csv \
    --batch_size 64 \
    --source_layer_idxs 1,2 \
    --target_layer_idxs 30,31 \
    --hf_token $HF_TOKEN_PATH \
    --run_evaluation --run_patchscopes_evaluation
```

## Code Structure

- `datasets`: contains datasets introduced in the two works.
  - `TwoHopFact.csv`
  - `SOCRATES.csv`
- `src`: contains the core functions.
  - `data_utils.py`, `model_utils.py`, `tokenization_utils.py` contain the common code used in both papers.
  - `inspection_utils.py` contains the code used in [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837).
  - `evaluation_utils.py` contains the code used in [Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](https://arxiv.org/abs/2411.16679).
  - `patchscopes_utils.py` contains the code used in the Patchscopes analysis in [Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](https://arxiv.org/abs/2411.16679).
- `results`: The result files from the experiments will be stored under this directory. This can be set by `--output_dir` argument.

## Citing our works

### [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837)

```
@inproceedings{
    yang2024latentreasoning,
    title={Do Large Language Models Latently Perform Multi-Hop Reasoning?},
    author={Sohee Yang and Elena Gribovskaya and Nora Kassner and Mor Geva and Sebastian Riedel},
    booktitle={Association for Computational Linguistics},
    year={2024},
    url={https://aclanthology.org/2024.acl-long.550}
}
```

### [Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](https://arxiv.org/abs/2411.16679)

```
@article{
    yang2024shortcutfree,
    title={Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?},
    author={Sohee Yang and Nora Kassner and Elena Gribovskaya and Sebastian Riedel and Mor Geva},
    journal={arXiv},
    year={2024},
    url={https://arxiv.org/abs/2411.16679}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

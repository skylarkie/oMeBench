# oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanistic Reasoning

**oMeBench** is a large-scale, expert-curated benchmark for evaluating large language models (LLMs) on **organic reaction mechanism reasoning**. It provides standardized datasets, mechanistic annotations, and a dynamic evaluation framework (**oMeS**) for measuring stepwise causal reasoning, intermediate generation, and chemical validity.

## Overview

Organic reaction mechanisms describe how reactants form intermediates and products through elementary steps. While LLMs can predict products or summarize reactions, they often fail to reason through these multi-step processes.

**oMeBench** addresses this gap by combining:

- **Expert-curated datasets** (`oMe-Gold`, `oMe-Template`, `oMe-Silver`)
- **Fine-grained step-level annotations** including types, subtypes, intermediates, and rationales
- **Dynamic evaluation** through the `oMeS` alignment-based scoring framework

## Repository Structure

```text
oMeBench/
├── data/
│   ├── oMe_Gold.json           # Expert-verified reactions
│   ├── oMe_Template.json       # Abstracted mechanistic templates
│   └── oMe_Silver.jsonl        # Expanded reactions
├── images/
│   └── score_vs_length_v2.png  # Benchmark visualization
├── prompts/
│   ├── default.txt             # Prompt template for direct evaluation
│   └── cot.txt                 # Prompt template for reasoning-style evaluation
├── scripts/
│   ├── run.py                  # Model evaluation script
│   └── utils_eval.py           # oMeS evaluation utilities
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset Summary

| Dataset | #Reactions | #Steps | #Types | #Subtypes | Description |
| --- | ---: | ---: | ---: | ---: | --- |
| **oMe-Gold** | 196 | 858 | 8 | 30 | Expert-verified reactions with natural-language rationales |
| **oMe-Template** | 167 | 722 | 8 | 30 | Generalized named-reaction templates with R-group placeholders |
| **oMe-Silver** | 2,493 | 10,541 | 8 | 30 | Expanded dataset for large-scale training and analysis |

Each reaction entry follows this general structure:

```json
{
  "reaction_id": "NR-201",
  "level": "medium",
  "name": "Nazarov Cyclization Reaction",
  "reactants_smiles": ["C(C)=CC(=O)C=C(C)", "CS(=O)(=O)O"],
  "products_smiles": ["CC1=CC(=O)CC1(C)"],
  "conditions": "H+ OSO2Me",
  "mechanism_step_nums": 4,
  "mechanism": [
    {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer", "intermediate_smiles": "C(C)=CC(=[OH+])C=CC"},
    {"step": 2, "type": "pericyclic", "subtype": "electrocyclization", "intermediate_smiles": "C1=CC(=O)CC1+"},
    {"step": 3, "type": "proton_transfer", "subtype": "acid_base_proton_transfer", "intermediate_smiles": "CC1=CC(=O)CC1"},
    {"step": 4, "type": "proton_transfer", "subtype": "acid_base_proton_transfer", "intermediate_smiles": "CC1=CC(=O)CC1(C)"}
  ]
}
```

## Evaluation Framework

The **oMeS** framework provides four complementary metrics:

| Metric | Description |
| --- | --- |
| **V** | SMILES validity: proportion of chemically valid predicted intermediates |
| **L** | Logical fidelity: step-type alignment score |
| **S_total** | Strict mechanistic score using exact type and structure matches |
| **S_partial** | Partial mechanistic score weighted by molecular similarity |

Mechanisms are aligned using a weighted Needleman-Wunsch algorithm with fingerprint-based similarity scoring. This allows partial credit for chemically plausible intermediates even when they are not exact matches.

## Quick Start

### 1. Install Dependencies

```bash
git clone <anonymous-repository-url>
cd oMeBench
pip install -r requirements.txt
```

### 2. Evaluate Models

Evaluate a predefined model:

```bash
python scripts/run.py --model gptoss
```

Evaluate a Hugging Face model or a local model path:

```bash
python scripts/run.py --custom-path "meta-llama/Meta-Llama-3-8B-Instruct"
python scripts/run.py --custom-path /path/to/model
```

Evaluate multiple predefined models:

```bash
python scripts/run.py --models gptoss mistral chemDFM
```

Add a suffix to output files:

```bash
python scripts/run.py --model gptoss --output-suffix experiment1
```

Use a local base model for LoRA adapters without hard-coding machine-specific paths:

```bash
OMEBENCH_BASE_MODEL_PATH=/path/to/base/model python scripts/run.py --custom-path /path/to/lora-adapter
```

## Citation

Citation metadata is intentionally omitted for anonymous review. It can be restored after the review process.

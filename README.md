# oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanistic Reasoning [[Paper](https://arxiv.org/abs/2510.07731)]

**oMeBench** is a large-scale, expert-curated benchmark designed to evaluate large language models (LLMs) on **organic reaction mechanism reasoning** — a core challenge in chemical intelligence.  
It provides standardized datasets, mechanistic annotations, and a dynamic evaluation framework (**oMeS**) that quantitatively measures model performance in stepwise causal reasoning, intermediate generation, and chemical validity.


## 🔬 Overview

Organic reaction mechanisms describe how reactants form intermediates and products through elementary steps.  
While LLMs can predict products or summarize reactions, they often fail to **reason** through these multi-step processes.  
**oMeBench** addresses this gap by combining:
- **Expert-curated datasets** (`oMe-Gold`, `oMe-Template`, `oMe-Silver`)
- **Fine-grained step-level annotations** (types, subtypes, intermediates, rationales)
- **Dynamic evaluation framework** (`oMeS`) for alignment-based scoring


## 📂 Repository Structure

```
oMeBench/
│
├── data/
│   ├── oMe_Gold.json           # Expert-verified, literature-curated reactions
│   ├── oMe_Template.json       # Abstracted mechanistic templates (R-group placeholders)
│   └── oMe_Silver.jsonl        # Expanded reactions via LLM-guided substitution
├── scripts/
│   ├── run.py         # Model evaluation script
│   └── utils_eval.py           # Evaluation utilities (oMeS framework)
├── prompts/
│   └── default_v2.txt          # Prompt template for model evaluation
├── LICENSE
└── README.md
```

## 🧪 Dataset Summary

| Dataset       | #Reactions | #Steps | #Types | #Subtypes | Description |
|----------------|-------------|---------|----------|-------------|--------------|
| **oMe-Gold** | 196 | 858 | 8 | 30 | Textbook-verified reactions with natural-language rationales |
| **oMe-Template** | 167 | 722 | 8 | 30 | Generalized named-reaction templates (R-group placeholders) |
| **oMe-Silver** | 2,493 | 10,541 | 8 | 30 | LLM-expanded dataset for large-scale training |

Each reaction entry includes:
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
    {"step": 1, "type": "proton_transfer", "intermediate": "C(C)=CC(=[OH+])C=CC"},
    {"step": 2, "type": "electrocyclization", "intermediate": "C1=CC(=O)CC1+"},
    {"step": 3, "type": "deprotonation", "intermediate": "CC1=CC(=O)CC1"},
    {"step": 4, "type": "tautomerization", "intermediate": "CC1=CC(=O)CC1(C)"}
  ]
}
```

## ⚖️ Evaluation Framework (oMeS)

The **oMeS** framework provides **four complementary metrics**:

| Metric        | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| **V**         | SMILES validity — proportion of chemically valid intermediates |
| **L**         | Logical fidelity — accuracy of step-type prediction            |
| **S_total**   | Strict mechanistic score (exact type + structure match)        |
| **S_partial** | Partial mechanistic score (weighted by molecular similarity)   |

Mechanisms are aligned using a **weighted Needleman–Wunsch algorithm** with fingerprint-based similarity scoring.
This allows partial credit for chemically plausible intermediates even if not identical.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/skylarkie/oMeBench.git
cd oMeBench
pip install -r requirements.txt
```

Required packages:
- `transformers` - For model loading and inference
- `torch` - PyTorch framework
- `peft` - For LoRA model support (optional)
- `rdkit` - For chemistry operations
- `tqdm` - For progress bars

### 2. Evaluate Models on oMeBench

The evaluation script `scripts/run.py` supports evaluating multiple types of models:

#### Single Model Evaluation
```bash
# Evaluate a predefined model
python scripts/run.py --model qwen3-4b-sft-1000

# Evaluate a custom model from local path
python scripts/run.py --custom-path /path/to/your/model

# Evaluate a Hugging Face model
python scripts/run.py --custom-path "meta-llama/Llama-3-8B-Instruct"
```

#### Batch Evaluation
```bash
# Evaluate multiple models in one run
python scripts/run.py --models qwen3-4b-sft-1000 llama3-8b-lora mistral-lora
```

#### Advanced Options
```bash
# Clear transformers cache before evaluation (useful for fixing loading issues)
python scripts/run.py --model qwen3-4b --clear-cache

# Add suffix to output files
python scripts/run.py --model qwen3-4b --output-suffix "experiment1"

# Clear cache only (no evaluation)
python scripts/run.py --clear-cache
```

## 🧩 Citation

If you use **oMeBench** or **oMeS**, please cite:

```
@misc{xu2025omebenchrobustbenchmarkingllms,
      title={oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning}, 
      author={Ruiling Xu and Yifan Zhang and Qingyun Wang and Carl Edwards and Heng Ji},
      year={2025},
      eprint={2510.07731},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.07731}, 
}
```
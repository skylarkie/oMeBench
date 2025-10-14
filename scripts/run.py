"""
Fine-tuned Model Evaluation Script for oMeBench

Solutions for model loading issues:
1. Multiple loading strategies: Try different parameter combinations on loading failure
2. Cache cleanup: Automatically clear transformers cache to resolve dynamic module issues
3. Path validation: Check model file existence before loading
4. Error recovery: Automatically retry with cache cleanup on loading failure

"""

import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
import torch
from utils_eval import oMeS, oMeSResult
from pathlib import Path
import shutil
import os

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT library not found. LoRA models will not be supported.")

def clear_transformers_cache():
    """Clear transformers cache to resolve dynamic module issues"""
    try:
        from transformers.utils import TRANSFORMERS_CACHE
        cache_dir = TRANSFORMERS_CACHE
    except ImportError:
        # Newer versions use HF_HOME
        cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    
    # Clear module cache that may cause issues
    modules_dir = os.path.join(cache_dir, 'modules')
    if os.path.exists(modules_dir):
        print(f"Clearing transformers modules cache: {modules_dir}")
        try:
            shutil.rmtree(modules_dir)
            print("Cache cleared successfully!")
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
    else:
        print("No transformers cache found to clear")

# === Available Model Configuration ===
AVAILABLE_MODELS = {
    # general models
    "gptoss": "openai/gpt-oss-20b",
    "llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    # chemistry models
    "chemDFM": "OpenDFM/ChemDFM-v1.5-8B",
    "olmo2": "allenai/OLMo-2-1124-13B-Instruct",
    "openbiollm": "aaditya/Llama3-OpenBioLLM-8B",
}

# Note: ether0 model has a dedicated run_ether0.py script for its special output format

# === Base Model Configuration (foundation models without chat template) ===
BASE_MODELS = {
    "llama3-8b-base",
    "qwen3-4b-base",
}

# === T5 Model Configuration (loaded with T5ForConditionalGeneration) ===
T5_MODELS = {
    "biot5",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on oMeBench")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=list(AVAILABLE_MODELS.keys()),
        help="Single model to evaluate (mutually exclusive with --models)"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Multiple models to evaluate (mutually exclusive with --model)"
    )
    parser.add_argument(
        "--custom-path", 
        type=str, 
        help="Custom model path (overrides predefined models)"
    )
    parser.add_argument(
        "--output-suffix", 
        type=str, 
        default="",
        help="Suffix to add to output files"
    )
    parser.add_argument(
        "--clear-cache", 
        action="store_true",
        help="Clear transformers cache before loading model (or only clear cache if no model specified)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.clear_cache and not args.model and not args.models:
        parser.error("--model or --models is required unless only using --clear-cache")
    
    # Check mutually exclusive arguments
    if args.model and args.models:
        parser.error("--model and --models are mutually exclusive")
    
    # Convert single model to list format
    if args.model:
        args.models = [args.model]
    
    return args

# === Path Configuration ===
ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "default.txt"
INPUT_PATH = ROOT / "data" / "oMe_Gold.json"
GOLD_PATH = ROOT / "data" / "oMe_Gold.json"

def is_huggingface_path(model_path: str):
    """Detect if the path is a Hugging Face model path"""
    # Hugging Face paths typically contain "/" but don't start with "/", and are not absolute paths
    return "/" in model_path and not os.path.isabs(model_path) and not os.path.exists(model_path)

def detect_model_type(model_path: str):
    """Detect model type: LoRA or full fine-tuning"""
    # For Hugging Face paths, default to full fine-tuning
    if is_huggingface_path(model_path):
        return "full"
    
    adapter_config_path = Path(model_path) / "adapter_config.json"
    return "lora" if adapter_config_path.exists() else "full"

def load_model(model_path: str, model_name: str, clear_cache_first: bool = False):
    """Load model from specified path (supports both LoRA and full fine-tuning)"""
    print(f"Loading model: {model_name}")
    print(f"Model path: {model_path}")
    
    # Check if it's a Hugging Face path
    is_hf_path = is_huggingface_path(model_path)
    
    if not is_hf_path:
        # Only check file existence for local paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Check if key files exist
        config_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        has_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in config_files)
        if not has_model_file:
            print(f"Warning: No standard model files found in {model_path}")
            print(f"Available files: {os.listdir(model_path)}")
    else:
        print(f"Detected Hugging Face model path: {model_path}")
    
    if clear_cache_first:
        clear_transformers_cache()
    
    model_type = detect_model_type(model_path)
    print(f"Detected model type: {model_type}")
    
    try:
        if model_type == "lora":
            return load_lora_model(model_path, model_name)
        else:
            return load_full_model(model_path, model_name)
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        
        # If cache hasn't been cleared, try clearing cache and retry
        if not clear_cache_first:
            print("Attempting to clear cache and retry...")
            clear_transformers_cache()
            try:
                if model_type == "lora":
                    return load_lora_model(model_path, model_name)
                else:
                    return load_full_model(model_path, model_name)
            except Exception as e2:
                print(f"Error loading model after cache clear: {e2}")
                raise e2
        else:
            raise e

def load_lora_model(model_path: str, model_name: str):
    """Load LoRA fine-tuned model"""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required for LoRA models. Install it with: pip install peft")
    
    print(f"Loading LoRA model from: {model_path}")
    
    # Check if it's a Hugging Face path (LoRA typically requires local paths)
    if is_huggingface_path(model_path):
        raise ValueError(f"LoRA models require local paths, but got Hugging Face path: {model_path}")
    
    # Read LoRA configuration
    config = PeftConfig.from_pretrained(model_path)
    base_model_name = config.base_model_name_or_path
    
    print(f"Base model: {base_model_name}")
    
    # Check if there's a local base model path
    local_base_path = "/work/hdd/bcei/yzhang66/huggingface/hub/models--allenai--OLMo-2-1124-13B-Instruct/snapshots/3a5c85baefbb1896a54d56fe2e76c0395627ddf4"
    if local_base_path and os.path.exists(local_base_path):
        print(f"Using local base model path: {local_base_path}")
        actual_base_path = local_base_path
    else:
        print(f"Using original base model path: {base_model_name}")
        actual_base_path = base_model_name
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        actual_base_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,  # Use local files only if using local path
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_path)
    
    # Load tokenizer (prefer LoRA path, fallback to base model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        print("LoRA path doesn't have tokenizer, using base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            actual_base_path, 
            trust_remote_code=True,
            local_files_only=True
        )
    
    print(f"LoRA model '{model_name}' loaded successfully!")
    return model, tokenizer

def load_full_model(model_path: str, model_name: str):
    """Load full fine-tuned model or base model"""
    print(f"Loading model from: {model_path}")
    
    # If it's a HF path and fully downloaded locally, resolve to snapshot path to avoid missing shard errors
    if is_huggingface_path(model_path):
        # Try to resolve existing local snapshot to avoid network access
        try:
            from huggingface_hub import snapshot_download
            local_snapshot = snapshot_download(model_path, local_files_only=True, ignore_patterns=["*.h5"], resume_download=False)
            print(f"[Info] Using local snapshot for {model_path}: {local_snapshot}")
            model_path = local_snapshot
        except Exception as e:
            # If snapshot is incomplete, notify and abort (prevent hanging in offline environment)
            raise RuntimeError(f"Local snapshot incomplete for {model_path}. Please download beforehand. Details: {e}")

    # Detect model type
    is_t5_model = model_name in T5_MODELS
    
    # Set appropriate parameters for base models
    base_kwargs = {
        "device_map": "auto",            # Map weights directly to target device, avoid secondary copy
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,        # Stream load by shard, speed up and reduce CPU RAM
    }
    
    # Use float16 for most models, but base models may need special handling
    if model_name not in BASE_MODELS:
        base_kwargs["torch_dtype"] = torch.float16
    
    # Try multiple loading strategies
    # For Hugging Face paths, prioritize online loading;
    # For local paths, prioritize local files to avoid unnecessary network requests.
    if is_huggingface_path(model_path):
        # First try pure offline loading; if shards are complete, avoid network requests; otherwise fallback to online mode.
        loading_strategies = [
            # Strategy 1: Use local cache only, fastest and no extra traffic
            {**base_kwargs, "local_files_only": True},
            # Strategy 2: Online loading (automatically completes if local is incomplete)
            {**base_kwargs},
            # Strategy 3: Force re-download to ensure complete weights
            {**base_kwargs, "force_download": True},
            # Strategy 4: Don't trust custom code
            {k: v for k, v in base_kwargs.items() if k != "trust_remote_code"},
            # Strategy 5: Fallback strategy
            {"device_map": "auto"},
        ]
    else:
        loading_strategies = [
            # Strategy 1: Use local files to avoid dynamic module issues
            {**base_kwargs, "local_files_only": True},
            # Strategy 2: Force re-download and trust remote code
            {**base_kwargs, "force_download": True},
            # Strategy 3: Don't use custom code
            {k: v for k, v in base_kwargs.items() if k != "trust_remote_code"},
            # Strategy 4: Basic loading strategy
            {"device_map": "auto"},
        ]
    
    # Disable gradient computation to save memory
    torch.set_grad_enabled(False)

    for i, kwargs in enumerate(loading_strategies, 1):
        try:
            print(f"Trying loading strategy {i}/{len(loading_strategies)}...")
            
            # Choose appropriate model class based on model type
            if is_t5_model:
                print(f"Loading T5 model with T5ForConditionalGeneration...")
                model = T5ForConditionalGeneration.from_pretrained(model_path, **kwargs)
            else:
                print(f"Loading causal LM model with AutoModelForCausalLM...")
                model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            
            print(f"Model loaded successfully with strategy {i}")
            break
        except Exception as e:
            print(f"Strategy {i} failed: {e}")
            if i == len(loading_strategies):
                print("All loading strategies failed!")
                raise e
            continue
    
    # Load tokenizer, also using multiple strategies
    tokenizer_strategies = [
        {"trust_remote_code": True, "local_files_only": True},
        {"trust_remote_code": True},
        {"local_files_only": True},
        {}
    ]
    
    for i, kwargs in enumerate(tokenizer_strategies, 1):
        try:
            print(f"Trying tokenizer loading strategy {i}/{len(tokenizer_strategies)}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            print(f"Tokenizer loaded successfully with strategy {i}")
            break
        except Exception as e:
            print(f"Tokenizer strategy {i} failed: {e}")
            if i == len(tokenizer_strategies):
                print("All tokenizer loading strategies failed!")
                raise e
            continue
    
    # Ensure tokenizer has necessary tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model '{model_name}' loaded successfully!")
    return model, tokenizer

# === Load Prompt Template ===
def load_prompt_template(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()

# === Replace Template Variables ===
def build_prompt(template: str, reactants, products, conditions):
    return template.replace("{ reactants_smiles }", json.dumps(reactants)) \
                   .replace("{ products_smiles }", json.dumps(products)) \
                   .replace("{ conditions }", conditions)

# === Detect Chat Template Support ===
def supports_chat_template(tokenizer, model_name: str):
    """Detect if model supports chat template"""
    # Check if in base model list
    if model_name in BASE_MODELS:
        return False
    
    # Check if tokenizer has chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        return True
    
    # Try applying chat template
    try:
        test_messages = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(test_messages, tokenize=False)
        return True
    except:
        return False

# === GPToss Special Handling Functions ===
def build_gptoss_prompt(prompt: str):
    """Build special format prompt for GPToss model"""
    return f"<|start|>user<|message|>{prompt}<|end|>\n<|start|>assistant<|channel|>final<|message|>"

def extract_gptoss_response(response: str):
    """Extract actual content from GPToss response"""
    # Find content after <|message|> tag
    if "<|message|>" in response:
        # Extract content after <|message|> until <|end|> or end of string
        start_idx = response.find("<|message|>") + len("<|message|>")
        end_idx = response.find("<|end|>", start_idx)
        if end_idx == -1:
            # If no <|end|>, take until end of string
            return response[start_idx:].strip()
        else:
            return response[start_idx:end_idx].strip()
    
    # If no <|message|> tag found, return original response
    return response.strip()

# === Build Prompt for Base Models ===
def build_base_model_prompt(prompt: str, model_name: str):
    """Build appropriate prompt format for base models"""
    if "llama" in model_name.lower():
        # LLaMA base model uses simple instruction format
        return f"### Instruction:\n{prompt}\n\n### Response:\n"
    elif "qwen" in model_name.lower():
        # Qwen base model uses simple format
        return f"User: {prompt}\nAssistant: "
    else:
        # Default format
        return f"{prompt}\n\nResponse: "

# === Generate Response Using Model ===
def generate_response(model, tokenizer, prompt: str, model_name: str = ""):
    try:
        # Check if it's GPToss model
        is_gptoss = "gptoss" in model_name.lower()
        
        if is_gptoss:
            # Use GPToss special format
            print(f"[Info] Using GPToss format for {model_name}")
            text = build_gptoss_prompt(prompt)
        else:
            # Detect chat template support
            use_chat_template = supports_chat_template(tokenizer, model_name)
            
            if use_chat_template:
                # Use chat template
                messages = [{"role": "user", "content": prompt}]
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"[Warning] Chat template failed, using base format: {e}")
                    text = build_base_model_prompt(prompt, model_name)
            else:
                # Use base format
                print(f"[Info] Using base model format for {model_name}")
                text = build_base_model_prompt(prompt, model_name)
        

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Check input length
        input_length = model_inputs.input_ids.shape[1]
        print(f"[Debug] Input length: {input_length} tokens")
        
        # Set pad_token if not available
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Generate response
        with torch.no_grad():

            generate_kwargs = {
                **model_inputs,
                "max_new_tokens": 2048,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            
            print(f"[Debug] Generation kwargs: {list(generate_kwargs.keys())}")
            generated_ids = model.generate(**generate_kwargs)
        
        # Decode response (only take newly generated part)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # If it's GPToss model, need special response handling
        if is_gptoss:
            response = extract_gptoss_response(response)
        
        return response.strip()
        
    except Exception as e:
        print(f"[Error] Generation failed for {model_name}: {e}")
        print(f"[Debug] Prompt length: {len(prompt)}")
        print(f"[Debug] Text length: {len(text) if 'text' in locals() else 'N/A'}")
        if 'model_inputs' in locals():
            print(f"[Debug] Input shape: {model_inputs.input_ids.shape if hasattr(model_inputs, 'input_ids') else 'N/A'}")
        import traceback
        traceback.print_exc()
        raise e

# === Generate Mechanisms One by One and Write to .jsonl ===
def generate_mechanisms(model, tokenizer, model_name: str, output_path: Path):
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    # If old output exists, read completed reaction_ids
    done_ids = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_ids.add(obj["reaction_id"])
                except:
                    continue

    prompt_template = load_prompt_template(PROMPT_PATH)

    with open(output_path, "a") as fout:
        for idx, reaction in tqdm(enumerate(data), total=len(data), desc=f"Generating with {model_name}"):
            rxn_id = reaction.get("reaction_id", f"rxn_{idx}")
            if rxn_id in done_ids:
                continue

            print(f"Processing reaction {idx+1}: {rxn_id}")
            prompt = build_prompt(prompt_template, reaction['reactants_smiles'], reaction['products_smiles'], reaction['conditions'])

            try:
                raw_output = generate_response(model, tokenizer, prompt, model_name)
                
                # Clean output format
                if "[ANSWER]" in raw_output:
                    # Handle [ANSWER] tag, whether or not [/ANSWER] exists
                    if "[/ANSWER]" in raw_output:
                        # answer is between [ANSWER] and [/ANSWER]
                        raw_output = raw_output.split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()
                    else:
                        # Only [ANSWER] at start, take content after it
                        raw_output = raw_output.split("[ANSWER]")[1].strip()
                    
                if raw_output.startswith("```"):
                    raw_output = raw_output.strip("`").replace("json\n", "").replace("JSON\n", "").strip()
                    
                if "is :" in raw_output:
                    raw_output = raw_output.split("is :")[1].strip()
                    
                
                # For base models, may need additional cleaning
                if model_name in BASE_MODELS:
                    # Remove possible instruction prefixes
                    raw_output = raw_output.replace("### Response:", "").replace("Assistant:", "").strip()
                    # Try to extract JSON part
                    import re
                    json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
                    if json_match:
                        raw_output = json_match.group(0)
                
                try:
                    parsed = json.loads(raw_output)
                except Exception as e:
                    print(f"[Warning] JSON parse failed on index {idx}: {e}")
                    print(f"[Debug] Raw output: {raw_output[:200]}...")
                    parsed = raw_output

                result = {
                    "reaction_id": rxn_id,
                    "input": {
                        "reactants_smiles": reaction['reactants_smiles'],
                        "products_smiles": reaction['products_smiles'],
                        "conditions": reaction['conditions']
                    },
                    "output": parsed
                }

            except Exception as e:
                print(f"[Error] Generation failed on index {idx}: {e}")
                result = {
                    "reaction_id": rxn_id,
                    "input": reaction,
                    "output": f"[Error] {e}"
                }

            fout.write(json.dumps(result) + "\n")
            fout.flush()

# === Evaluate Mechanism Results ===
def evaluate_results(model_name: str, model_path: str, output_path: Path, eval_path: Path):
    with open(output_path, "r") as f:
        answers = [json.loads(line) for line in f]
    with open(GOLD_PATH, "r") as f:
        gold_data = json.load(f)

    answer_map = {a["reaction_id"]: a["output"] for a in answers}
    result = [{"model": model_name, "model_path": model_path, "total_predictions": len(answers)}]

    total_evaluated = 0
    successful_evaluations = 0

    for gold_item in gold_data:
        rxn_id = gold_item['reaction_id']
        pred_output = answer_map.get(rxn_id)
        
        total_evaluated += 1
        
        # Handle dict format output (some models may return dict with mechanism field)
        if isinstance(pred_output, dict):
            pred_output = pred_output.get("mechanism", pred_output)

        # If output is not a list, record 0 score
        if not isinstance(pred_output, list):
            print(f"[Error] Invalid or missing output for {rxn_id}, scoring as 0")
            result.append({
                "reaction_id": rxn_id,
                "S_total": 0.0,
                "S_partial": 0.0,
                "V": 0,
                "L": 0,
                "alignment": []
            })
            continue

        # Try to evaluate valid prediction output
        try:
            pred = [(j['subtype'], j['intermediate_smiles']) for j in pred_output]
            gold = [(j['subtype'], j['intermediate_smiles'], j['step_weight']) for j in gold_item['mechanism']]
            res = oMeS(gold, pred)
            result.append({
                "reaction_id": rxn_id,
                "S_total": round(res.S_total, 2),
                "S_partial": round(res.S_partial, 2),
                "V": res.V,
                "L": res.L,
                "alignment": res.alignment
            })
            successful_evaluations += 1
        except Exception as e:
            print(f"[Error] Evaluation failed for {rxn_id}: {e}")
            # Record 0 score on evaluation failure
            result.append({
                "reaction_id": rxn_id,
                "S_total": 0.0,
                "S_partial": 0.0,
                "V": 0,
                "L": 0,
                "alignment": []
            })

    # Calculate overall statistics
    if total_evaluated > 0:
        s_total_scores = [r["S_total"] for r in result[1:] if "S_total" in r]
        s_partial_scores = [r["S_partial"] for r in result[1:] if "S_partial" in r]
        v_scores = [r["V"] for r in result[1:] if "V" in r]
        l_scores = [r["L"] for r in result[1:] if "L" in r]
        
        result[0].update({
            "evaluated_reactions": total_evaluated,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": total_evaluated - successful_evaluations,
            "avg_S_total": round(sum(s_total_scores) / len(s_total_scores), 3) if s_total_scores else 0,
            "avg_S_partial": round(sum(s_partial_scores) / len(s_partial_scores), 3) if s_partial_scores else 0,
            "avg_V": round(sum(v_scores) / len(v_scores), 3) if v_scores else 0,
            "avg_L": round(sum(l_scores) / len(l_scores), 3) if l_scores else 0,
        })

    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Evaluation results saved to {eval_path}")
    print(f"Model: {model_name}")
    print(f"Total reactions evaluated: {total_evaluated}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations (scored as 0): {total_evaluated - successful_evaluations}")
    if total_evaluated > 0:
        print(f"Average S_total: {result[0].get('avg_S_total', 'N/A')}")
        print(f"Average S_partial: {result[0].get('avg_S_partial', 'N/A')}")
        print(f"Average V: {result[0].get('avg_V', 'N/A')}")
        print(f"Average L: {result[0].get('avg_L', 'N/A')}")

# === Process Single Model ===
def process_single_model(model_name: str, args):
    """Process evaluation for a single model"""
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")
    
    # Determine model path
    if args.custom_path:
        model_path = args.custom_path
        full_model_name = f"custom_{Path(args.custom_path).name}"
    else:
        model_path = AVAILABLE_MODELS[model_name]
        full_model_name = model_name
    
    # Add suffix
    if args.output_suffix:
        full_model_name += f"_{args.output_suffix}"
    
    # Set output paths
    output_path = ROOT / "test_result_cot" / f"{full_model_name}.jsonl"
    eval_path = ROOT / "eval_result_cot" / f"{full_model_name}.json"
    
    # Ensure directories exist
    output_path.parent.mkdir(exist_ok=True)
    eval_path.parent.mkdir(exist_ok=True)
    
    try:
        # Load model
        model, tokenizer = load_model(model_path, full_model_name, args.clear_cache)
        
        # Generate predictions
        generate_mechanisms(model, tokenizer, full_model_name, output_path)
        
        # Evaluate results
        evaluate_results(full_model_name, model_path, output_path, eval_path)
        
        print(f"✅ Successfully completed evaluation for {full_model_name}")
        
    except Exception as e:
        print(f"❌ Error processing {full_model_name}: {e}")
        print(f"Skipping to next model...")

# === Main Pipeline ===
def main():
    args = parse_args()
    
    # If user only wants to clear cache
    if args.clear_cache and not args.models:
        print("Clearing transformers cache...")
        clear_transformers_cache()
        print("Cache cleared. Exiting.")
        return
    
    # Process multiple models
    print(f"Starting evaluation for {len(args.models)} model(s): {', '.join(args.models)}")
    
    for i, model_name in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Processing {model_name}...")
        process_single_model(model_name, args)
    
    print(f"\n{'='*60}")
    print(f"All {len(args.models)} model(s) processing completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


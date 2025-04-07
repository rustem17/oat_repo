import torch
import os
import re
import json
import argparse
import sys
import tempfile
import subprocess
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from tqdm import tqdm

from src import EleutherSparseAutoencoder

def load_local_model(baseline=False):
    """loads the local model with optional LoRA adapter"""
    if baseline:
        print("loading baseline llama 3 model")
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
        print("baseline model and tokenizer ready")
        return encoder
    else:
        print("loading modified llama 3 model")
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)

        repo_id = "Mechanistic-Anomaly-Detection/llama3-oat-generation-linear"
        local_dir = "llama3-oat-generation-linear"

        print(f"downloading snapshot from {repo_id} to {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)

        if not isinstance(encoder.model, PeftModel):
            print("merging LoRA adapters")
            encoder.model = PeftModel.from_pretrained(encoder.model, local_dir)
            encoder.model = encoder.model.merge_and_unload()

        print("modified model and tokenizer ready")
        return encoder

def extract_python_code(text):
    """extracts python code from model output"""
    code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()

    lines = text.split('\n')
    code_lines = []
    start_found = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(("def ", "class ", "import ", "from ")):
             start_found = True
        if start_found:
            code_lines.append(line)
        elif len(code_lines) > 0 and not stripped_line and len(lines) > lines.index(line) + 1 and not lines[lines.index(line) + 1].strip().startswith((" ", "\t", "#")):
            break

    if code_lines:
        return "\n".join(code_lines).strip()

    return text

def check_correctness(problem_id, generated_code, test_code, timeout=10):
    """executes generated code against test cases"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        file_path = tmp_file.name
        full_code = f"{generated_code}\n\n{test_code}"
        tmp_file.write(full_code)

    result = {'passed': False, 'error': None}
    process = None
    python_executable = sys.executable
    try:
        process = subprocess.run(
            [python_executable, file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=os.environ.copy(),
        )
        if process.returncode == 0:
            result['passed'] = True
        else:
            error_output = f"exit code: {process.returncode}\n"
            if process.stdout: error_output += f"stdout:\n{process.stdout}\n"
            if process.stderr: error_output += f"stderr:\n{process.stderr}\n"
            result['error'] = error_output.strip()

    except subprocess.TimeoutExpired:
        result['error'] = f"execution timed out after {timeout} seconds"
    except Exception as e:
        result['error'] = f"execution failed: {str(e)}"
    finally:
        if process and process.poll() is None:
             try: process.terminate()
             except Exception: pass
             try: process.wait(timeout=1)
             except Exception: pass
             if process.poll() is None:
                 try: process.kill()
                 except Exception: pass
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except Exception: pass

    return result

def generate_response(encoder, prompt_text, max_new_tokens=512):
    """generates a response from the model"""
    messages = [{"content": prompt_text, "role": "user"}]
    tokens = encoder.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(encoder.model.device)

    with torch.no_grad():
        output = encoder.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0
        )

    decoded = encoder.tokenizer.decode(output[0])
    return decoded

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="use baseline model without adapter")
    parser.add_argument("--output_dir", type=str, default="eval_results_mbpp", help="output directory for results")
    parser.add_argument("--num_examples", type=int, default=None, help="number of examples to evaluate (default: all)")
    return parser

def evaluate_mbpp(encoder, output_dir, num_examples=None, baseline=False):
    """evaluates model on mbpp dataset"""
    print("loading mbpp dataset")
    mbpp_dataset = load_dataset("mbpp", "sanitized", split="test")
    
    if num_examples:
        mbpp_dataset = mbpp_dataset.select(range(min(num_examples, len(mbpp_dataset))))
    
    results_file = os.path.join(output_dir, "mbpp_results.json")
    summary_file = os.path.join(output_dir, "mbpp_summary.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    existing_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
            print(f"loaded {len(existing_results)} existing results")
        except:
            pass
    
    completed_ids = {r["task_id"] for r in existing_results}
    results = existing_results

    print(f"evaluating {len(mbpp_dataset)} mbpp problems")
    for i, problem in enumerate(tqdm(mbpp_dataset)):
        task_id = str(problem["task_id"])
        
        if task_id in completed_ids:
            continue
            
        prompt = f"Write a Python function based on this description:\n\n{problem['text']}\n\n```python\n"
        
        try:
            response = generate_response(encoder, prompt)
            extracted_code = extract_python_code(response)
            
            test_code = problem['test_setup_code'] + "\n" + "\n".join(problem['test_list'])
            
            eval_result = check_correctness(
                task_id,
                extracted_code,
                test_code,
                timeout=20
            )
            
            result = {
                "task_id": task_id,
                "prompt": prompt,
                "response": response,
                "extracted_code": extracted_code,
                "passed": eval_result['passed'],
                "error": eval_result['error']
            }
            
            results = [r for r in results if r['task_id'] != task_id]
            results.append(result)
            completed_ids.add(task_id)
            
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"error processing problem {task_id}: {e}")
    
    total = len(results)
    passed = sum(1 for r in results if r.get('passed', False))
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nmbpp evaluation results:")
    print(f"total: {total}")
    print(f"passed: {passed}")
    print(f"pass rate: {pass_rate:.2f}%")

def main():
    args = create_parser().parse_args()
    
    encoder = load_local_model(baseline=args.baseline)
    evaluate_mbpp(encoder, args.output_dir, args.num_examples, args.baseline)

if __name__ == "__main__":
    main()
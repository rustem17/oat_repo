import argparse
import json
import os

import torch
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

from automated_redteaming.online_attacks import PromptAutomaticIterativeRefinement, GreedyCoordinateGradient
from automated_redteaming.strongreject_evaluator import autograde_response
from src.encoders import EleutherSparseAutoencoder, DeepmindSparseAutoencoder
from src.utils import load_probes

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_file", type=str, default="attack_results.json", help="file to save results")
    parser.add_argument("--model_type", type=str, choices=["llama3", "gemma2"], default="llama3", help="model type")
    parser.add_argument("--probe_path", type=str, default="llama3-oat-generation-linear", help="path to probe weights")
    parser.add_argument("--repo_id", type=str, default="Mechanistic-Anomaly-Detection/llama3-oat-generation-linear", help="hf repo id for probe weights")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--attack_type", type=str, default="pair", choices=["pair", "gcg"], help="attack type to use")
    parser.add_argument("--num_examples", type=int, default=10, help="number of examples to attack")
    parser.add_argument("--calibration_examples", type=int, default=5, help="number of examples for calibration")
    
    return parser

def get_probe_scores(probes, encoder, examples, batch_size=16, max_length=2048, probe_layers=None):
    """get probe scores for examples, returns {layer_idx: [[(token, score), ...], ...]}"""
    if probe_layers is None:
        probe_layers = list(probes.keys())
    
    results = {layer: [] for layer in probe_layers}
    
    for example in examples:
        tokens = encoder.tokenizer(
            example,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).input_ids.to(encoder.model.device)
        
        with torch.no_grad():
            outputs = encoder.model(
                input_ids=tokens,
                output_hidden_states=True
            )
            
        for layer in probe_layers:
            layer_activations = outputs.hidden_states[layer + 1][0]  # +1 because hidden_states includes embedding layer
            probe = probes[layer]
            scores = probe.predict(layer_activations).cpu()
            
            token_scores = []
            for i, score in enumerate(scores):
                token = encoder.tokenizer.decode(tokens[0, i])
                token_scores.append((token, score.item()))
            
            results[layer].append(token_scores)
    
    return results

def calculate_threshold(probe_scores, percentile=99):
    """calculate threshold value at 1% fpr (99th percentile)"""
    all_scores = []
    for layer_scores in probe_scores.values():
        for token_scores in layer_scores:
            scores = [score for _, score in token_scores]
            all_scores.extend(scores)
    return np.percentile(all_scores, percentile)

def main(args):
    print(f"running redteaming with {args.attack_type} attack on {args.model_type} model")
    
    # load model and tokenizer
    if args.model_type == "llama3":
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    else:
        encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
    
    # download probes if needed
    local_dir = args.probe_path
    if not os.path.exists(local_dir):
        print(f"downloading probes from {args.repo_id} to {local_dir}")
        snapshot_download(repo_id=args.repo_id, local_dir=local_dir)
    
    # load probes and apply LoRA if available
    if not isinstance(encoder.model, PeftModel) and os.path.exists(local_dir):
        print(f"loading LoRA adapter from {local_dir}")
        encoder.model = PeftModel.from_pretrained(encoder.model, local_dir)
        encoder.model = encoder.model.merge_and_unload()
    
    # load probes
    probe_file = os.path.join(local_dir, "probes.pt")
    if os.path.exists(probe_file):
        print(f"loading probes from {probe_file}")
        probes = load_probes(probe_file)
    else:
        print(f"warning: no probes found at {probe_file}")
        probes = None
    
    # load dataset
    dataset = load_dataset("abhayesian/augmented_advbench_v5")["train"].shuffle()
    
    # calibrate with examples
    print(f"calculating threshold using {args.calibration_examples} examples")
    calibration_examples = []
    for i in range(args.calibration_examples):
        prompt = dataset[i]["prompt"]
        calibration_examples.append(prompt)
    
    # get probe scores for calibration
    probe_scores = get_probe_scores(
        probes=probes,
        encoder=encoder,
        examples=calibration_examples,
        batch_size=16,
        max_length=8192,
        probe_layers=[12],
    )
    
    # calculate threshold at 99th percentile (1% fpr)
    threshold = calculate_threshold(probe_scores)
    print(f"calculated probe threshold value: {threshold}")
    
    # initialize attack
    if args.attack_type == "pair":
        print("initializing PAIR attack")
        attack = PromptAutomaticIterativeRefinement(
            model=encoder.model,
            tokenizer=encoder.tokenizer,
            api_key=openai_api_key,
        )
    elif args.attack_type == "gcg":
        print("initializing GCG attack")
        attack = GreedyCoordinateGradient(
            model=encoder.model,
            tokenizer=encoder.tokenizer,
        )
    
    results = []
    
    # process examples
    num_examples = min(args.num_examples, len(dataset) - args.calibration_examples)
    print(f"\nevaluating {num_examples} examples...")
    
    for i in range(args.calibration_examples, args.calibration_examples + num_examples):
        try:
            prompt = dataset[i]["prompt"]
            target = " ".join(dataset[i]["rejected"].split()[:10])
            chosen_response = dataset[i]["chosen"]
            
            print(f"\nprocessing example {i}:")
            print(f"prompt: {prompt[:100]}...")
            
            # execute attack
            if args.attack_type == "pair":
                top_candidate, high_score, response = attack.compute_adversary(prompt, target)
                attack_dict = {
                    "prompt": prompt,
                    "refusal": chosen_response,
                    "jailbreak": top_candidate,
                    "model_response": response,
                    "pair_score": high_score,
                }
            elif args.attack_type == "gcg":
                top_candidate, gcg_loss, response = attack.compute_adversary(prompt, target)
                attack_dict = {
                    "prompt": prompt,
                    "refusal": chosen_response,
                    "jailbreak": top_candidate,
                    "model_response": response,
                    "gcg_loss": gcg_loss,
                }
            
            # get strongreject score
            try:
                if openai_api_key:
                    strongreject_score = autograde_response(prompt, response).score_zero_to_one
                    print(f"strongreject score: {strongreject_score:.4f}")
                    attack_dict["strongreject_score"] = float(strongreject_score)
                else:
                    print("no openai api key, skipping strongreject evaluation")
                    attack_dict["strongreject_score"] = None
            except Exception as e:
                print(f"error getting strongreject score: {e}")
                attack_dict["strongreject_score"] = None
            
            # get probe scores
            if probes:
                probe_scores = get_probe_scores(
                    probes=probes,
                    encoder=encoder,
                    examples=[top_candidate],
                    batch_size=16,
                    max_length=8192,
                    probe_layers=[12],
                )
                
                # check if any token's score exceeds threshold
                token_scores = probe_scores[12][0]
                max_probe_score = max(score for _, score in token_scores)
                
                if max_probe_score > threshold:
                    print(f"BLOCKED - probe score {max_probe_score:.4f} exceeded threshold {threshold:.4f}")
                    attack_dict["blocked"] = True
                else:
                    print(f"ALLOWED - response: {response[:100]}...")
                    attack_dict["blocked"] = False
                
                # save probe metrics
                attack_dict.update({
                    "probe_threshold": float(threshold),
                    "max_probe_score": float(max_probe_score),
                    "probe_token_scores": [[token, float(score)] for token, score in token_scores]
                })
            
            # add to results
            results.append(attack_dict)
            
            # save after each example
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"error processing example {i}: {e}")
            continue
    
    print(f"\nfinished evaluating {len(results)} examples")
    print(f"results saved to {args.output_file}")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
import os
import json
import re
import time
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset

# -- LLaMA local model imports & setup ----------------------------------------
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from src import EleutherSparseAutoencoder, load_probes  # Adjust if needed

# -----------------------------------------------------------------------------


def load_local_llama_model(baseline=False):
    """
    Loads the local LLaMA-based model.
    If baseline is False, we load the modified LLaMA model (merging LoRA adapters);
    if baseline is True, we load a baseline model without merging LoRA (or any modifications).
    Returns: (encoder) which includes a .model and a .tokenizer.
    """
    if baseline:
        print("Loading baseline LLaMA 3 model locally...")
        # For your baseline, you might wish to load without any modifications.
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
        # Optionally: perform any baseline-specific post-processing here.
        print("Baseline model and tokenizer ready.")
        return encoder
    else:
        print("Loading modified LLaMA 3 model locally...")
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)

        repo_id = "Mechanistic-Anomaly-Detection/llama3-oat-generation-linear"
        local_dir = "llama3-oat-generation-linear"

        print(f"Downloading snapshot from {repo_id} to {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
        )

        # If not already a PeftModel, merge LoRA adapters.
        if not isinstance(encoder.model, PeftModel):
            print("Merging LoRA adapters...")
            encoder.model = PeftModel.from_pretrained(encoder.model, local_dir)
            encoder.model = encoder.model.merge_and_unload()

        linear_probes = load_probes(os.path.join(local_dir, "probes.pt"))
        print("Modified model and tokenizer ready.")
        return encoder



def generate_local_llama_response(encoder, prompt_text):
    """
    Generates a response from the local LLaMA model using your
    instructed chat template and a fixed 'assistant' generation style.

    Args:
      encoder: the object returned by load_local_llama_model()
      prompt_text: string containing the question or combined instruction.

    Returns:
      A string containing the model's decoded text output.
    """
    tokens = encoder.tokenizer.apply_chat_template(
        [{"content": prompt_text, "role": "user"}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).cuda()

    with torch.no_grad():
        output = encoder.model.generate(
            tokens,
            max_new_tokens=400,
            do_sample=False,
            temperature=0.0,
            top_p=1.0
        )

    decoded = encoder.tokenizer.decode(output[0])
    return decoded


def generate_local_llama_response_with_mask(encoder, prompt_text):
    tokens = encoder.tokenizer.apply_chat_template(
        [{"content": prompt_text, "role": "user"}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(tokens, dict):
        tokens = {k: v.cuda() for k, v in tokens.items()}
    else:
        tokens = tokens.cuda()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

    # Prepare input_ids and (if available) attention_mask.
    if isinstance(tokens, dict):
        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")
    else:
        input_ids = tokens
        attention_mask = None

    with torch.no_grad():
        output = encoder.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=400,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=encoder.tokenizer.eos_token_id
        )

    if output.dim() == 1:
        decoded = encoder.tokenizer.decode(output.tolist())
    else:
        decoded = encoder.tokenizer.decode(output[0].tolist())
    return decoded


# -- MMLU-Pro evaluation code ------------------------------------------------
def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(split_data):
    """
    Reformat the MMLU-Pro data so we can index it by category.
    Removes any 'N/A' option from the options array.
    """
    res_df = []
    for each in split_data:
        options = [opt for opt in each["options"] if opt != "N/A"]
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def extract_answer(text):
    """
    Attempts to extract an answer A-J from the LLaMA output
    by matching patterns like:
      'the answer is (X)' or 'Answer: X' or final mention of A-J
    """
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # fallback
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(encoder, single_question, dev_examples_dict, exist_result):
    """
    For a single question without chain-of-thought examples, we:
      1) Check if result already cached in exist_result.
      2) Build a simpler prompt that directly asks for the final answer.
      3) Call the local LLaMA model and extract the predicted option.
    """
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], True

    category = single_question["category"]
    question = single_question["question"]
    options = single_question["options"]

    # Build a simplified prompt without chain-of-thought examples:
    prompt = ("You will be given a multiple choice question about {}. "
              "Provide the final answer in the format: \"The answer is (X)\".\n\n").format(category)

    input_text = "Question: {}\nOptions:\n".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        input_text += f"{choice_map[i]}. {opt}\n"
    input_text += "Answer: "  # No extra chain-of-thought text here.

    full_prompt = prompt + input_text

    try:
        # response = generate_local_llama_response(encoder, full_prompt)
        response = generate_local_llama_response_with_mask(encoder, full_prompt)
    except Exception as e:
        print("Error while generating response:", e)
        return None, None, False

    pred = extract_answer(response)
    return pred, response, False


def update_result(output_res_path):
    """
    Loads existing results from output_res_path and calculates the
    per-category stats. Returns (res_list, category_record).
    """
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                for each in res:
                    category = each["category"]
                    if category not in category_record:
                        category_record[category] = {"corr": 0.0, "wrong": 0.0}
                    # If no pred, random fallback for the sake of completeness
                    if not each.get("pred"):
                        random.seed(12345)
                        x = random.randint(0, len(each["options"]) - 1)
                        if x == each["answer_index"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
                    else:
                        if each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error reading or updating results:", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    """
    Merges the current question's result into the overall res array
    to avoid duplication. If question_id is found, update it; else append.
    """
    merged = False
    for i, single in enumerate(res):
        if (single["question_id"] == curr["question_id"]
            and single["question"] == curr["question"]):
            res[i] = curr
            merged = True
            break
    if not merged:
        res.append(curr)
    return res


def save_res(res, output_res_path):
    """
    Cleans duplicates and saves result JSON.
    """
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(temp))


def save_summary(category_record, output_summary_path):
    """
    Computes overall accuracy and per-category accuracy, then writes JSON summary.
    """
    total_corr = 0.0
    total_wrong = 0.0
    for cat, stats in category_record.items():
        if cat == "total":
            continue
        cat_acc = stats["corr"] / (stats["corr"] + stats["wrong"]) if (stats["corr"]+stats["wrong"])>0 else 0
        category_record[cat]["acc"] = cat_acc
        total_corr += stats["corr"]
        total_wrong += stats["wrong"]
    acc = total_corr / (total_corr + total_wrong) if (total_corr+total_wrong) > 0 else 0
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


def evaluate(encoder, subjects, output_dir, baseline=False):
    """
    The main evaluation loop. Goes through each subject in MMLU-Pro test set,
    uses the dev set for few-shot, calls local LLaMA, extracts the answer,
    and saves the results incrementally.
    """
    test_df, dev_df = load_mmlu_pro()

    if not subjects:
        subjects = list(test_df.keys())
    print("Evaluating on subjects:", subjects)

    for subject in subjects:
        test_data = test_df[subject]
        if baseline:
            res_suffix = "_baseline_result.json"
            sum_suffix = "_baseline_summary.json"
        else:
            res_suffix = "_result.json"
            sum_suffix = "_summary.json"

        output_res_path = os.path.join(output_dir, subject + res_suffix)
        output_summary_path = os.path.join(output_dir, subject + sum_suffix)

        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data, desc=f"Evaluating subject {subject}"):
            label = each["answer"]
            pred, response, exist = single_request(encoder, each, dev_df, res)

            if not exist and response is not None:
                res, category_record = update_result(output_res_path)
                if subject not in category_record:
                    category_record[subject] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)

                if pred == label:
                    category_record[subject]["corr"] += 1
                else:
                    category_record[subject]["wrong"] += 1

                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)

                res, category_record = update_result(output_res_path)

        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


# -- Main script entry point --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    os.makedirs("eval_results_mmlu", exist_ok=True)
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results_mmlu/",
                        help="Directory to store results.")
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all",
                        help="Comma-separated list of subjects or 'all'.")
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load your local LLaMA model
    encoder = load_local_llama_model(baseline=True)

    # Run the MMLU-Pro evaluation
    evaluate(encoder, assigned_subjects, args.output_dir, baseline=True)

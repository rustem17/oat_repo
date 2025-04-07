import json
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from tqdm.auto import tqdm
import wandb

from .utils import convert_seconds_to_time_str, get_valid_token_mask
from .attacks import train_attack, benchmark_obfuscated_softprompt


class Probe(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # implements a forward pass through the probe
        raise NotImplementedError

    def compute_loss(self, acts, labels, mask=None):
        # computes loss between probe outputs and labels
        # acts: shape (d1, d2, ..., dn, d_model) - model activations
        # labels: shape (d1, d2, ..., dn) - binary labels
        # mask: optional boolean mask for selecting positions

        logits = self.forward(acts)

        if mask is not None:
            if mask.shape != logits.shape:
                mask = mask.view(logits.shape)

            logits = logits[mask]
            labels = labels[mask]

        # BCE loss
        labels = labels.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

    def predict(self, x):
        # x should be of shape (d1, d2, ..., dn, d_model)
        # should return a tensor of shape (d1, d2, ..., dn)
        return torch.sigmoid(self.forward(x))


def train_online_probe(
    encoder,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    lora_params={},
    adversarial_training=False,
    probe_lr=1e-3,
    adapter_lr=5e-5,
    kl_penalty=1e-2,
    max_length=1024,
    n_steps=1000,
    n_steps_per_logging=100,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    pretrained_probes=None,
    only_return_on_tokens_between=None,
    only_choose_prompt_tokens_between=None,
    only_probe_tokens_between=None,  # mask for selecting which tokens to probe
    epsilon=10.0,
    adversary_lr=1e-3,
    pgd_iterations=32,
    clip_grad_norm=1.0,
    start_adv_training_at_step=1024,
    freeze_probes_during_adversarial_training=True,
    freeze_lora_during_warmup=False,
    use_lora_adapter=True,
    run_softprompt_eval_every=128,
    softprompt_evals_data={},
    **kwargs,
):
    # trains probes on model activations with optional adversarial training and lora adapters
    # if hasattr(encoder, 'model_name'):
    #     model_name = encoder.model_name
    # elif hasattr(encoder.model, 'config'):
    #     model_name = encoder.model.config._name_or_path
    # else:
    #     model_name = type(encoder.model).__name__

    # probe_type = "linear" if "Linear" in create_probe_fn.__name__ else "nonlinear"

    assert n_grad_accum == 0 or n_steps % n_grad_accum == 0

    probes = {}
    optimizers = {}
    if pretrained_probes is not None:
        print("Using pretrained probes...")
        probes = pretrained_probes
    else:
        probes = {layer: create_probe_fn() for layer in layers}

    optimizers = {
        layer: torch.optim.AdamW(probe.parameters(), lr=probe_lr)
        for layer, probe in probes.items()
    }

    probes = {layer: probe.to(device) for layer, probe in probes.items()}

    if use_lora_adapter:
        lora_model = initialize_lora_adapter(encoder, layers, lora_params)
        adapter_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=adapter_lr)
    else:
        lora_model = encoder.model
        adapter_optimizer = None

    encoder.tokenizer.padding_side = "right"
    positive_tokens = encoder.tokenizer(
        positive_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    positive_input_ids = positive_tokens["input_ids"]
    positive_attention_mask = positive_tokens["attention_mask"]
    negative_tokens = encoder.tokenizer(
        negative_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    negative_input_ids = negative_tokens["input_ids"]
    negative_attention_mask = negative_tokens["attention_mask"]

    print("positive attention_mask:", positive_attention_mask)
    print("negative attention mask:", negative_attention_mask)

    print("only_return_on_tokens_between=", only_return_on_tokens_between)

    if only_return_on_tokens_between is not None:
        zero_positive_mask = get_valid_token_mask(
            positive_input_ids, only_return_on_tokens_between
        )
        zero_negative_mask = get_valid_token_mask(
            negative_input_ids, only_return_on_tokens_between
        )
    else:
        zero_positive_mask = torch.ones_like(positive_input_ids).bool()
        zero_negative_mask = torch.ones_like(negative_input_ids).bool()

    print("Mask for positive example (sum):", zero_positive_mask[0].sum().item())
    print("Mask for negative example (sum):", zero_negative_mask[0].sum().item())

    print("only_probe_tokens_between=", only_probe_tokens_between)

    if only_probe_tokens_between is not None:
        probe_positive_mask = get_valid_token_mask(
            positive_input_ids, only_probe_tokens_between
        )
        probe_negative_mask = get_valid_token_mask(
            negative_input_ids, only_probe_tokens_between
        )
    else:
        probe_positive_mask = zero_positive_mask
        probe_negative_mask = zero_negative_mask

    print("only_choose_prompt_tokens_between=", only_choose_prompt_tokens_between)

    if only_choose_prompt_tokens_between is not None:
        assert adversarial_training
        pos_only_choose_mask = get_valid_token_mask(
            positive_input_ids, only_choose_prompt_tokens_between
        )
        pos_only_choose_mask = pos_only_choose_mask.to(device)
    else:
        pos_only_choose_mask = None

    n_examples = min(len(positive_examples), len(negative_examples))

    continue_training_next_epoch = True
    current_step = 0
    start_time = time.time()

    accumulated_toward_pgd_loss = 0
    accumulated_probe_pgd_loss = 0
    accumulated_probe_loss = 0
    accumulated_kl_loss = 0
    steps_since_last_log = 0
    info = {
        "softprompt_evals": [],
    }

    wrappers = []
    adversaries = []
    pgd_probe_loss = 0

    pbar = tqdm(total=n_steps, desc="Training LORA+Probe")

    while continue_training_next_epoch:
        perm = torch.randperm(n_examples)

        for i in range(0, n_examples, batch_size):
            if i + batch_size > n_examples:
                break

            batch_perm = perm[i : i + batch_size]
            pos_batch_input_ids = positive_input_ids[batch_perm].to(device)
            pos_batch_attention_mask = positive_attention_mask[batch_perm].to(device)
            neg_batch_input_ids = negative_input_ids[batch_perm].to(device)
            neg_batch_attention_mask = negative_attention_mask[batch_perm].to(device)
            pos_batch_zero_mask = zero_positive_mask[batch_perm].to(device).bool()
            neg_batch_zero_mask = zero_negative_mask[batch_perm].to(device).bool()
            pos_batch_probe_mask = probe_positive_mask[batch_perm].to(device).bool()
            neg_batch_probe_mask = probe_negative_mask[batch_perm].to(device).bool()

            # print("Positive input IDs:", positive_input_ids[0])
            # print("Decoded text:", encoder.tokenizer.decode(positive_input_ids[0]))

            # print("Special token new line:", encoder.tokenizer.convert_tokens_to_ids("\n"))
            # print("Special token user:", encoder.tokenizer.convert_tokens_to_ids("user"))
            # print("Special token model:", encoder.tokenizer.convert_tokens_to_ids("model"))
            # print("Special token <start_of_turn>:", encoder.tokenizer.convert_tokens_to_ids("<start_of_turn>"))
            # print("Special token <end_of_turn>:", encoder.tokenizer.convert_tokens_to_ids("<end_of_turn>"))

            # print("Special token <start_of_turn>user:", encoder.tokenizer.convert_tokens_to_ids("<start_of_turn>user"))
            # print("Special token <end_of_turn>:", encoder.tokenizer.convert_tokens_to_ids("<end_of_turn>"))
            # print("Special token <start_of_turn>model:", encoder.tokenizer.convert_tokens_to_ids("<start_of_turn>model"))

            if pos_only_choose_mask is not None:
                pos_batch_only_choose_mask = (
                    pos_only_choose_mask[batch_perm].to(device).bool()
                )

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if adversarial_training and current_step >= start_adv_training_at_step:

                    if current_step == start_adv_training_at_step:
                        print("FORMATTING EXAMPLES FOR ADVERSARIAL TRAINING")
                        print("This is the target mask:")
                        print(
                            repr(
                                encoder.tokenizer.decode(
                                    pos_batch_input_ids[0][pos_batch_zero_mask[0]]
                                )
                            )
                        )
                        print("This is the prompt mask:")
                        print(
                            repr(
                                encoder.tokenizer.decode(
                                    pos_batch_input_ids[0][
                                        pos_batch_only_choose_mask[0]
                                    ]
                                )
                            )
                        )
                        print("This is the probe mask:")
                        print(
                            repr(
                                encoder.tokenizer.decode(
                                    pos_batch_input_ids[0][pos_batch_probe_mask[0]]
                                )
                            )
                        )

                    disable_model_gradients(lora_model)

                    if (
                        current_step - start_adv_training_at_step
                    ) % run_softprompt_eval_every == 0:
                        print("RUNNING SOFTPROMPT EVALUATION")
                        results = benchmark_obfuscated_softprompt(
                            model=lora_model,
                            model_layers_module="base_model.model.model.layers",
                            tokenizer=encoder.tokenizer,
                            probes=probes,
                            **softprompt_evals_data,
                        )

                        info["softprompt_evals"].append(results)
                        print("Jailbreak Success Rate:", results["jailbreak_rate"])
                        print("Average positive score:", results["avg_positive_score"])
                        print("Average negative score:", results["avg_negative_score"])

                    losses, wrappers = train_attack(
                        adv_tokens=pos_batch_input_ids,
                        prompt_mask=pos_batch_only_choose_mask,
                        target_mask=pos_batch_zero_mask,
                        model=lora_model,
                        tokenizer=encoder.tokenizer,
                        model_layers_module="base_model.model.model.layers",
                        layer=["embedding"],
                        epsilon=epsilon,
                        learning_rate=adversary_lr,
                        pgd_iterations=pgd_iterations,
                        probes=probes,
                        probe_mask=pos_batch_probe_mask,  # Pass probe mask
                        adversary_type="pgd",
                    )

                    pgd_toward_loss = losses["toward"]
                    pgd_probe_loss = losses["probe"]

                    enable_model_gradients(lora_model)
                else:
                    pgd_toward_loss = (
                        0  # Set to 0 when adversarial training is not used
                    )
                    pgd_probe_loss = 0
                    wrappers = []

                for wrapper in wrappers:
                    wrapper.enabled = True

                pos_output = lora_model(
                    input_ids=pos_batch_input_ids,
                    output_hidden_states=True,
                )
                pos_acts = {
                    layer: pos_output.hidden_states[layer + 1] for layer in layers
                }

            pos_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    print(f"[DEBUG] step={current_step} (POS) layer={layer}")
                    print("  pos_acts[layer] shape:", pos_acts[layer].shape)
                    print("  pos_batch_probe_mask shape:", pos_batch_probe_mask.shape)
                    print("  probe_mask sum:", pos_batch_probe_mask.sum().item())

                    logits = probe.forward(pos_acts[layer])
                    print("  logits shape:", logits.shape)
                    print("  logits min:", float(logits.min().item()),
                            " logits max:", float(logits.max().item()))

                    pos_targets = torch.ones_like(
                        pos_acts[layer][..., 0], device=device
                    )
                    pos_layer_loss = probe.compute_loss(
                        pos_acts[layer],
                        pos_targets,
                        mask=pos_batch_probe_mask,  # Use probe mask
                    )
                    pos_loss += pos_layer_loss

            pos_loss.backward(retain_graph=True)

            for wrapper in wrappers:
                wrapper.enabled = False

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                neg_output = lora_model(
                    input_ids=neg_batch_input_ids,
                    output_hidden_states=True,
                )
                neg_logits = neg_output.logits
                neg_acts = {
                    layer: neg_output.hidden_states[layer + 1] for layer in layers
                }

            neg_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    print(f"[DEBUG] step={current_step} (NEG) layer={layer}")
                    print("  neg_acts[layer] shape:", neg_acts[layer].shape)
                    print("  neg_batch_probe_mask shape:", neg_batch_probe_mask.shape)
                    print("  probe_mask sum:", neg_batch_probe_mask.sum().item())

                    logits = probe.forward(neg_acts[layer])
                    print("  logits shape:", logits.shape)
                    print("  logits min:", float(logits.min().item()),
                            " logits max:", float(logits.max().item()))
                    neg_targets = torch.zeros_like(
                        neg_acts[layer][..., 0], device=device
                    )
                    neg_layer_loss = probe.compute_loss(
                        neg_acts[layer],
                        neg_targets,
                        mask=neg_batch_probe_mask,  # Use probe mask
                    )
                    neg_loss += neg_layer_loss

            neg_loss.backward(retain_graph=True)

            # KL divergence of logits from base model logits
            with torch.no_grad():
                lora_model.disable_adapter_layers()
                base_neg_output = lora_model(
                    input_ids=neg_batch_input_ids,
                    # attention_mask=neg_batch_attention_mask,
                )
                lora_model.enable_adapter_layers()

            base_logits = base_neg_output.logits[neg_batch_zero_mask]
            model_logits = neg_logits[neg_batch_zero_mask]

            kl_loss = F.kl_div(
                F.log_softmax(base_logits, dim=-1),
                F.softmax(model_logits, dim=-1),
                reduction="batchmean",
            )

            # Backward pass on KL divergence
            (kl_loss / (kl_loss.detach() + 1e-8) * kl_penalty).backward()

            # Accumulate losses
            accumulated_probe_loss += pos_loss.item() + neg_loss.item()
            accumulated_kl_loss += kl_loss.item()
            accumulated_toward_pgd_loss += (
                pgd_toward_loss if adversarial_training else 0
            )
            accumulated_probe_pgd_loss += pgd_probe_loss if adversarial_training else 0
            steps_since_last_log += 1

            if (i // batch_size + 1) % n_grad_accum == 0 or (
                i + batch_size
            ) >= n_examples:

                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_model.parameters(), clip_grad_norm
                    )
                    all_probe_params = [
                        param
                        for probe in probes.values()
                        for param in probe.parameters()
                    ]
                    torch.nn.utils.clip_grad_norm_(all_probe_params, clip_grad_norm)

                if not freeze_probes_during_adversarial_training or not (
                    adversarial_training and current_step > start_adv_training_at_step
                ):
                    for optimizer in optimizers.values():
                        optimizer.step()
                        optimizer.zero_grad()

                if not freeze_lora_during_warmup or not (
                    adversarial_training and current_step < start_adv_training_at_step
                ):
                    if adapter_optimizer is not None:
                        adapter_optimizer.step()
                        adapter_optimizer.zero_grad()

            current_step += 1

            if current_step % n_steps_per_logging == 0:
                avg_probe_loss = accumulated_probe_loss / steps_since_last_log
                avg_kl_loss = accumulated_kl_loss / steps_since_last_log
                avg_toward_pgd_loss = (
                    accumulated_toward_pgd_loss / steps_since_last_log
                    if adversarial_training
                    else 0
                )
                avg_probe_pgd_loss = (
                    accumulated_probe_pgd_loss / steps_since_last_log
                    if adversarial_training
                    else 0
                )
                avg_total_loss = avg_probe_loss + avg_kl_loss

                metrics = {
                    "train/total_loss": avg_total_loss,
                    "train/probe_loss": avg_probe_loss,
                    "train/kl_loss": avg_kl_loss,
                    "step": current_step,
                }

                if adversarial_training:
                    metrics.update({
                        "train/pgd_toward_loss": avg_toward_pgd_loss,
                        "train/pgd_probe_loss": avg_probe_pgd_loss,
                    })

                wandb.log(metrics)

                log_message = (
                    f"Step: {current_step}/{n_steps}, "
                    f"Time: {convert_seconds_to_time_str(time.time() - start_time)}, "
                    f"Avg Total Loss: {avg_total_loss:.4f}, "
                    f"Avg Probe Loss: {avg_probe_loss:.4f}, "
                    f"Avg KL Loss: {avg_kl_loss:.4f}"
                )

                if adversarial_training:
                    log_message += f", Avg Toward PGD Loss: {avg_toward_pgd_loss:.4f}"
                    log_message += f", Avg Probe PGD Loss: {avg_probe_pgd_loss:.4f}"

                print(log_message)

                accumulated_toward_pgd_loss = 0
                accumulated_probe_pgd_loss = 0
                accumulated_probe_loss = 0
                accumulated_kl_loss = 0
                steps_since_last_log = 0

            if current_step >= n_steps:
                continue_training_next_epoch = False
                break

            pbar.update(1)  # Update progress bar

    try:
        if adversarial_training:
            wandb.run.summary.update({
                "final_pgd_toward_loss": avg_toward_pgd_loss,
                "final_pgd_probe_loss": avg_probe_pgd_loss,
            })
    except Exception as e:
        print("oops, problem with unbound variables")

    return probes, lora_model, info


def save_probes(probes, save_path):
    # saves trained probes to disk
    torch.save(probes, save_path)


def load_probes(load_path):
    # loads probes from disk
    return torch.load(load_path, weights_only=False)


def initialize_lora_adapter(encoder, layers, lora_params):
    # initializes lora adapter for parameter-efficient fine-tuning
    # freezes base model parameters and adds trainable lora layers
    for param in encoder.model.parameters():
        param.requires_grad = False

    r = lora_params.get("r", 64)
    alpha = lora_params.get("alpha", 128)
    dropout = lora_params.get("dropout", 0.0)
    bias = lora_params.get("bias", "none")
    target_modules = lora_params.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        layers_to_transform=list(range(0, max(layers) + 1)),
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(encoder.model, lora_config)

    return lora_model


def disable_model_gradients(lora_model):
    for param in lora_model.parameters():
        param.requires_grad_(False)


def enable_model_gradients(lora_model):
    n_layers = lora_model.config.num_hidden_layers
    for i in range(n_layers):
        for name, param in lora_model.get_submodule("base_model.model.model.layers")[
            i
        ].named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

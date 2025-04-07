import os
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# token and mask utilities
def get_valid_token_mask(tokens, only_return_on_tokens_between):
    # creates a mask for tokens between specified start and end markers
    # supports both token ids and predicate functions as markers
    if tokens.dim() not in (1, 2):
        raise ValueError("Input tensor must be 1D or 2D")

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    batch_size, seq_length = tokens.shape
    start_token, end_token = only_return_on_tokens_between

    def match(seq_idx, token, tokens, matcher):
        if callable(matcher):
            return matcher(seq_idx, token.item(), tokens)
        else:
            return token.item() == matcher

    # Initialize the mask with zeros
    mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=tokens.device)

    for i in range(batch_size):
        include_indices = False
        for j in range(seq_length):
            token = tokens[i, j]
            if match(j, token, tokens[i], start_token):
                include_indices = True
            elif match(j, token, tokens[i], end_token):
                include_indices = False
            elif include_indices:
                mask[i, j] = True

    return mask.squeeze(0) if tokens.dim() == 1 else mask

# layer activation extraction
def get_all_residual_acts_unbatched(
    model, input_ids, attention_mask=None, only_return_layers=None
):
    # extracts residual activations from specified layers
    # returns dictionary of layer_idx -> activation tensor
    
    # get total layers in model
    num_layers = model.config.num_hidden_layers

    # filter to valid requested layers
    layers_to_return = (
        set(range(num_layers))
        if only_return_layers is None
        else set(only_return_layers)
    )
    layers_to_return = {layer for layer in layers_to_return if 0 <= layer < num_layers}

    # Forward pass with output_hidden_states=True to get all hidden states
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # The hidden states are typically returned as a tuple, with one tensor per layer
    # We want to exclude the embedding layer (first element) and get only the residual activations
    hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer

    # Extract and return only the required layers
    return {layer: hidden_states[layer] for layer in layers_to_return}

# hook utilities for activation extraction
def extract_submodule(module, target_path):
    # retrieves a submodule from the model using a dot-separated path
    if not target_path:
        return module

    # navigate through nested modules
    path_parts = target_path.split(".")
    current_module = module
    for part in path_parts:
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            raise AttributeError(f"Module has no attribute '{part}'")
    return current_module

def forward_pass_with_hooks(model, input_ids, hook_points, attention_mask=None):
    # performs forward pass and captures activations at specified hook points
    # returns dictionary of hook_name -> activation tensor
    activations = {}

    # create hook function for each hook point
    def create_hook(hook_name):
        def hook_fn(module, input, output):
            # handle transformer block outputs (which are tuples)
            if type(output) == tuple:
                output = output[0]
            assert isinstance(output, torch.Tensor)  # verify we got tensor activations
            activations[hook_name] = output

        return hook_fn

    # Add a hook to every submodule we want to cache
    hooks = []
    for hook_point in hook_points:
        submodule = extract_submodule(model, hook_point)
        hook = submodule.register_forward_hook(create_hook(hook_point))
        hooks.append(hook)
    try:

        # Perform the forward pass
        with torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:

        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return activations

# model loading utilities
def load_hf_model_and_tokenizer(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
    tokenizer_name=None,
    requires_grad=False,
):
    # loads huggingface model and tokenizer with common configurations
    # Auto-select attention implementation based on model
    if "gpt2" in model_name or "gemma" in model_name:
        attn_implementation = "eager"

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()

    # Disable model grad if we're not training
    if not requires_grad:
        model.requires_grad_(False)

    # Get the tokenizer name if not specified
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id  # Make sure the eos in the generation config is the same as the tokenizer
    
    return model, tokenizer

# time formatting utility
def convert_seconds_to_time_str(seconds):
    # converts seconds to human-readable time string (e.g. "2h 3m 45s")
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = seconds // 3600
        seconds = seconds % 3600
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds}s"
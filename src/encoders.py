import warnings
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from .utils import load_hf_model_and_tokenizer


class ModelWrapper:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def format_inputs(self, prompt, system_prompt=None):
        # Format input for language models using the tokenizer's chat template.
        def format_single_input(single_prompt):
            # Format a single prompt with optional system message using the tokenizer's chat template.
            # Prepare messages in the format expected by chat models
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add user prompt
            messages.append({"role": "user", "content": single_prompt})
            try:

                # Use the tokenizer's chat template
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except AttributeError:
                warnings.warn(
                    "The provided tokenizer does not have the 'apply_chat_template' method. "
                    "Falling back to a simple format. This may not be optimal for all models.",
                    UserWarning,
                )

                # Simple fallback format
                formatted = ""
                if system_prompt:
                    formatted += f"{system_prompt}\n"
                formatted += f"{single_prompt}"

                # Add BOS token if available
                bos_token = getattr(self.tokenizer, "bos_token", "")
                return f"{bos_token}{formatted}"

        # Handle input based on whether it's a single string or a list of strings
        if isinstance(prompt, str):
            return format_single_input(prompt)
        elif isinstance(prompt, list):
            return [format_single_input(p) for p in prompt]
        else:
            raise ValueError("prompt must be either a string or a list of strings")

    def sample_generations(
        self,
        prompts,
        format_inputs=True,
        batch_size=4,
        system_prompt=None,
        **generation_kwargs,
    ):
        # Make sure prompts is a list
        if not isinstance(prompts, list):
            print(f"Prompts must be a list but is {type(prompts)}, wrapping in a list")
            prompts = [prompts]

        # Format inputs if requested
        if format_inputs:
            prompts = self.format_inputs(prompts, system_prompt)
        all_generations = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize inputs
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_return_sequences=1,
                    **generation_kwargs,
                )

            # Decode generated text
            batch_generations = self.tokenizer.batch_decode(outputs)
            batch_generations = [
                gen.replace(self.tokenizer.pad_token, "") for gen in batch_generations
            ]
            all_generations.extend(batch_generations)

        return all_generations


class LanguageModelWrapper(ModelWrapper):
    # Wrapper class for language models
    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
    
    def get_model_residual_acts(self, examples, batch_size=8, max_length=1024, use_memmap=None, only_probe_tokens_between=None, **kwargs):
        """
        Get model residual activations for a batch of examples.
        This is a compatibility method required by the probe_training.py code.
        
        Args:
            examples: List of text examples to get activations for
            batch_size: Batch size for processing examples
            max_length: Maximum sequence length
            use_memmap: If not None, path to save activations as memory-mapped arrays
            only_probe_tokens_between: Optional token range to filter activations
            
        Returns:
            Dictionary of layer activations
        """
        from .utils import get_all_residual_acts, get_valid_token_mask
        
        # Tokenize examples
        tokenized = self.tokenizer(
            examples, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        
        # Get all residual activations from the model
        inputs = tokenized["input_ids"].to(self.model.device)
        attention_mask = tokenized["attention_mask"].to(self.model.device)
        
        # Get residual activations for all layers by default
        all_activations = get_all_residual_acts(
            self.model, 
            inputs, 
            attention_mask=attention_mask,
            batch_size=batch_size
        )
        
        # Apply token mask if specified
        if only_probe_tokens_between is not None:
            for layer, activations in all_activations.items():
                mask = get_valid_token_mask(inputs, only_probe_tokens_between)
                # Apply mask to each position in the sequence
                for i in range(len(inputs)):
                    for j in range(inputs.shape[1]):
                        if not mask[i, j]:
                            activations[i, j] = 0
        
        return all_activations


class LlamaModelLoader:
    @staticmethod
    def load_llama3_model(
        instruct=True,
        other_model_tokenizer=(None, None),
        device="cuda",
    ):
        # Load the model from huggingface
        if other_model_tokenizer[0] is not None:
            print("A model and tokenizer were provided, using those instead")
            model, tokenizer = other_model_tokenizer
        else:
            model_name = (
                "meta-llama/Meta-Llama-3-8B-Instruct"
                if instruct
                else "meta-llama/Meta-Llama-3-8B"
            )
            model, tokenizer = load_hf_model_and_tokenizer(
                model_name, device_map=device
            )

        return LanguageModelWrapper(
            model=model,
            tokenizer=tokenizer,
        )


class PythiaModelLoader:
    @staticmethod
    def load_pythia_model(
        model_size="160m",
        deduped=True,
        other_model_tokenizer=(None, None),
        device="cuda",
    ):
        # Loading Pythia model
        assert model_size in [
            "70m",
            "160m",
        ], "Only 70m and 160m models are supported"

        # Load the model from huggingface
        if other_model_tokenizer[0] is not None:
            print("A model and tokenizer were provided, using those instead")
            model, tokenizer = other_model_tokenizer
        else:
            model_name = (
                f"EleutherAI/pythia-{model_size}-deduped"
                if deduped
                else f"EleutherAI/pythia-{model_size}"
            )
            model, tokenizer = load_hf_model_and_tokenizer(model_name)

        return LanguageModelWrapper(
            model=model,
            tokenizer=tokenizer,
        )


class GemmaModelLoader:
    @staticmethod
    def load_gemma2_model(
        instruct=True,
        other_model_tokenizer=(None, None),
        device="cuda",
    ):
        # Loading Gemma 2 model
        if other_model_tokenizer[0] is not None:
            print("A model and tokenizer were provided, using those instead")
            model, tokenizer = other_model_tokenizer
        else:
            model_name = "google/gemma-2-9b-it" if instruct else "google/gemma-2-9b"
            model, tokenizer = load_hf_model_and_tokenizer(model_name, device_map=device)

        return LanguageModelWrapper(
            model=model,
            tokenizer=tokenizer,
        )


class QwenCoderLoader:
    @staticmethod
    def load_qwen_coder(
        model_name=None, 
        device="cuda", 
        trust_remote_code=True, 
        load_in_8bit=False
    ):
        """
        Convenience loader for Qwen/Qwen2.5-Coder (or Qwen-7B, etc.).
        If no model_name is provided, defaults to an instruct coder.

        :param model_name: e.g. "Qwen/Qwen2.5-Coder-7B-Instruct"
        :param device: "cuda" or "cpu"
        :param trust_remote_code: Needed if transformers < 4.37
        :param load_in_8bit: Optional, for memory optimization (requires bitsandbytes)
        """
        if model_name is None:
            model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # If using bitsandbytes 8-bit:
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                quantization_config=bnb_config,
                device_map="auto"  # for multi-GPU, or "cuda:0" if single GPU
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            model.to(device)

        # If tokenizer doesn't have pad token, set it to eos_token
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            warnings.warn("No pad token set for Qwen, so using eos_token as pad_token.")

        return LanguageModelWrapper(model, tokenizer)
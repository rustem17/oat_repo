"""
Embedding-based attacks that optimize tokens to be appended to prompts.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import wandb
from redglass.activations import Activations
from redglass.data import TargetedDialogueDataset, TokenizedDataset
from redglass.detectors import DirectionDetector
from redglass.log import logger
from redglass.types import Dialogue
from redglass.utils import preprocess_dialogue


@dataclass
class AttackResult:
    """
    The result of an attack.

    Args:
        optimized_embeddings: The optimized embeddings. If the attack is not universal,
            this will be a list of tensors, one for each example in the dataset.
        losses: The losses for each example in each epoch
    """

    optimized_embeddings: torch.Tensor | list[torch.Tensor]
    losses: list[list[float]]


class EmbeddingAttack:
    """
    An attack that optimizes a set of embeddings that are appended to prompts to make
    the model generate specific target completions.

    Attributes:
        model: The model to attack
        tokenizer: The tokenizer for the model
        num_tokens: Number of optimization tokens to append
        learning_rate: Learning rate for optimization
        num_iterations_per_example: Number of optimization iterations per example
        device: Device to run optimization on
        embedding_matrix: The optimized embeddings (filled after optimize() is called)
        token_ids: The token IDs corresponding to the optimized embeddings
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        universal: bool,
        num_tokens: int = 50,
        optimization_init: str | None = None,
        learning_rate: float = 0.1,
        num_iterations_per_example: int = 10,
        num_epochs: int = 1,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        detectors: list[DirectionDetector] = [],
        detector_loss_weight: float = 0.1,
        verbose: bool = False,
    ):
        """
        Initialize the embedding attack.

        Args:
            model: The model to attack
            tokenizer: The tokenizer for the model
            universal: Whether to use a universal initialization for the embeddings
            optimization_init: The initial optimization tokens. If None, random
                initialization is used.
            num_tokens: Number of optimization tokens to append. If `optimization_init` is
                not None, this is ignored.
            learning_rate: Learning rate for optimization
            num_iterations_per_example: Number of optimization iterations per example in
                the dataset.
            num_epochs: Number of epochs to run the optimization for.
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: The name of the Weights & Biases project
            detectors: List of detectors to use for detection
            detector_loss_weight: Weight for the detector loss
            verbose: Whether to print verbose output
        """
        self.model = model
        self.tokenizer = tokenizer
        self.universal = universal
        self.optimization_init = optimization_init
        self.num_tokens = num_tokens
        self.learning_rate = learning_rate
        self.num_iterations_per_example = num_iterations_per_example
        self.num_epochs = num_epochs

        self.device = self.model.device
        if self.device == "cpu":
            logger.warning(
                "Running optimization on CPU. This may be very slow. GPU is recommended."
            )

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        self.optim_token = "<|optim-loc|>"
        self.optimized_embeddings = None  # filled after optimize() is called

        self.detectors = detectors
        self.detector_loss_weight = detector_loss_weight

        self.verbose = verbose

    def _prepare_dataset(
        self, dataset: TargetedDialogueDataset
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the dataset for optimization by creating prompt dialogues and tokenizing.

        Args:
            dataset: The dataset to prepare

        Returns:
            Tuple of (tokenized_prompts, tokenized_targets)
        """
        prompt_dialogues = []
        for i, dialogue in enumerate(dataset.dialogues):
            # Create a copy of the dialogue without the last assistant message
            prompt_dialogue = []
            for msg in dialogue:
                if msg is dialogue[-1] and msg.role == "assistant":
                    # Skip the last assistant message as this is what we want to generate
                    continue
                prompt_dialogue.append(msg)

            prompt_dialogue[-1].content += self.optim_token
            prompt_dialogues.append(Dialogue(prompt_dialogue))

        # Preprocess and format prompts
        processed = [preprocess_dialogue(d) for d in prompt_dialogues]
        formatted_dialogues: list[str] = self.tokenizer.apply_chat_template(
            processed, tokenize=False, add_generation_prompt=True
        )  # type: ignore

        # Tokenize before optimization token
        before_dialogues = [d.split(self.optim_token)[0] for d in formatted_dialogues]
        tokenized_before = self.tokenizer(
            before_dialogues,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        ).to(self.device)

        # Tokenize after optimization token
        after_dialogues = [d.split(self.optim_token)[1] for d in formatted_dialogues]
        tokenized_after = self.tokenizer(
            after_dialogues,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        # Tokenize targets
        tokenized_targets = self.tokenizer(
            dataset.target_completions,
            return_tensors="pt",
            padding=True,
            padding_side="right",  # since we're adding to the end of the sequence
            add_special_tokens=False,  # not add <begin_of_text>
        ).to(self.device)

        return tokenized_before, tokenized_after, tokenized_targets

    def _print_input_and_target(
        self,
        before_tokens: torch.Tensor,
        after_tokens: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        Print the input and target tokens.
        """
        print("=" * 100)
        print(self.tokenizer.decode(before_tokens[0]) + self.optim_token)
        print("-" * 100)
        print(self.tokenizer.decode(after_tokens[0]))
        print("-" * 100)
        print(self.tokenizer.decode(target[0]))
        print("=" * 100)

    def _generate_and_print_output(
        self,
        before_embeddings: torch.Tensor,
        after_embeddings: torch.Tensor,
        optimized_embeddings: torch.Tensor,
    ):
        """
        Generate and print the output of the model.
        """
        full_embeddings = torch.cat(
            [before_embeddings, optimized_embeddings, after_embeddings], dim=1
        )
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=full_embeddings,
                max_new_tokens=100,
            )
        print("=" * 100)
        print(self.tokenizer.decode(outputs[0]))
        print("=" * 100)

    def initialize_optimized_embeddings(
        self,
    ) -> Tuple[torch.Tensor, torch.optim.Optimizer]:
        """
        Initialize the optimized embeddings.
        """

        logger.debug("Initializing new optimized embeddings")
        embedding_layer = self.model.get_input_embeddings()
        if self.optimization_init is None:
            optimized_embeddings = torch.randn(
                (1, self.num_tokens, self.model.config.hidden_size),
                device=self.device,
                dtype=self.model.dtype,
            ).requires_grad_()
        else:
            optimized_tokens = self.tokenizer(
                self.optimization_init,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)
            optimized_embeddings = (
                embedding_layer(optimized_tokens).detach().clone().requires_grad_()
            )
        optimizer = torch.optim.Adam(
            [optimized_embeddings], lr=self.learning_rate, eps=1e-6
        )

        return optimized_embeddings, optimizer

    def optimize(self, dataset: TargetedDialogueDataset) -> AttackResult:
        """
        Optimize embedding tokens for the given dataset.
        Creates individual optimized embeddings for each example.

        Args:
            dataset: Dataset containing prompt-target pairs

        Returns:
            AttackResult containing the optimized embeddings and losses
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        if self.use_wandb:
            wandb.init(project=self.wandb_project)

        embedding_layer = self.model.get_input_embeddings()
        dataset_size = len(dataset)
        tokenized_before, tokenized_after, tokenized_targets = self._prepare_dataset(
            dataset
        )

        if self.universal:
            optimized_embeddings, optimizer = self.initialize_optimized_embeddings()

        epoch_losses = []
        optimized_embeddings_dict = {}
        for epoch in range(self.num_epochs):
            example_losses = []
            for idx in tqdm(range(dataset_size), desc="Example"):
                if not self.universal:
                    optimized_embeddings, optimizer = (
                        self.initialize_optimized_embeddings()
                    )

                before_tokens = (
                    tokenized_before.input_ids[idx].to(self.device).unsqueeze(0)
                )
                before_attention_mask = (
                    tokenized_before.attention_mask[idx].to(self.device).unsqueeze(0)
                )
                before_embeddings = embedding_layer(before_tokens)

                after_tokens = (
                    tokenized_after.input_ids[idx].to(self.device).unsqueeze(0)
                )
                after_attention_mask = (
                    tokenized_after.attention_mask[idx].to(self.device).unsqueeze(0)
                )
                after_embeddings = embedding_layer(after_tokens)

                # Tokenize target
                target = tokenized_targets.input_ids[idx].to(self.device).unsqueeze(0)
                target_attention_mask = (
                    tokenized_targets.attention_mask[idx].to(self.device).unsqueeze(0)
                )
                target_embeddings = embedding_layer(target)

                if self.verbose:
                    self._print_input_and_target(before_tokens, after_tokens, target)

                for i in range(self.num_iterations_per_example):
                    optimizer.zero_grad()

                    combined_embeddings = torch.cat(
                        [
                            before_embeddings.detach(),
                            optimized_embeddings,
                            after_embeddings.detach(),
                            target_embeddings.detach(),
                        ],
                        dim=1,
                    )
                    combined_attention_mask = torch.cat(
                        [
                            before_attention_mask.detach(),
                            torch.ones(
                                (1, optimized_embeddings.shape[1]), device=self.device
                            ),
                            after_attention_mask.detach(),
                            target_attention_mask.detach(),
                        ],
                        dim=1,
                    )

                    outputs = self.model(
                        inputs_embeds=combined_embeddings,
                        attention_mask=combined_attention_mask,
                        output_hidden_states=True,
                    )
                    logits = outputs.logits

                    # Shift logits so token n-1 predicts token n
                    target_len = target.shape[1]
                    shift_logits = logits[:, -target_len - 1 : -1, :]
                    shift_labels = target

                    ce_loss = torch.nn.functional.cross_entropy(
                        shift_logits.transpose(1, 2),
                        shift_labels,
                        ignore_index=self.tokenizer.pad_token_id,
                    )

                    if self.detectors is not None:
                        tokenized_dataset = TokenizedDataset(
                            dialogues=[],
                            formatted_dialogues=[],
                            str_tokens=[],
                            tokens=torch.zeros_like(combined_attention_mask).int(),
                            attention_mask=combined_attention_mask.bool(),
                            detection_mask=combined_attention_mask.int(),
                        )
                        hidden_states = outputs.hidden_states

                        detector_losses = []
                        for detector in self.detectors:
                            detector_layers = detector.layers
                            all_acts = torch.stack(
                                [
                                    hidden_states[layer_idx]
                                    for layer_idx in detector_layers
                                ],
                                dim=2,
                            )
                            activations = Activations(
                                all_acts=all_acts,
                                tokenized_dataset=tokenized_dataset,
                                layers=detector_layers,
                            )
                            scores = detector.score(activations, all_acts=True)
                            detector_losses.append(
                                torch.cat([s for s in scores.scores])
                            )

                        detector_loss = torch.mean(torch.stack(detector_losses))

                    loss = (
                        ce_loss + self.detector_loss_weight * detector_loss
                        if self.detectors
                        else ce_loss
                    )
                    ce_loss_float = ce_loss.item()
                    detector_loss_float = (
                        detector_loss.item() if self.detectors else 0.0
                    )
                    loss_float = loss.item()
                    if self.verbose:
                        logger.info(
                            f"Example {idx}, Iteration {i:3d} | CE: {ce_loss_float:.4f} | Detector: {detector_loss_float:.4f}"
                        )
                    example_losses.append(loss_float)

                    loss.backward()
                    optimizer.step()

                epoch_losses.append(example_losses)
                logger.debug(f"Example {idx} | Loss: {example_losses[-1]}")
                optimized_embeddings_dict[idx] = optimized_embeddings
                if self.verbose:
                    self._generate_and_print_output(
                        before_embeddings, after_embeddings, optimized_embeddings
                    )
                if self.use_wandb:
                    wandb.log({"loss": example_losses[-1]})

        if self.universal:
            # return last optimized embedding
            self.optimized_embeddings = optimized_embeddings
            return AttackResult(
                optimized_embeddings=optimized_embeddings,
                losses=epoch_losses,
            )
        else:
            # sort optimized embeddings by example index and return list of tensors
            optimized_embeddings_list = sorted(
                optimized_embeddings_dict.items(), key=lambda x: x[0]
            )
            optimized_embeddings = [x[1] for x in optimized_embeddings_list]
            return AttackResult(
                optimized_embeddings=optimized_embeddings,
                losses=epoch_losses,
            )

    def generate_with_optimized_embeddings(
        self, dataset: TargetedDialogueDataset
    ) -> list[str]:
        """
        Generate completions with the optimized embeddings.
        """
        if self.optimized_embeddings is None:
            raise ValueError(
                "Optimized embeddings not found. Please run optimize() with universal=True first."
            )

        pass

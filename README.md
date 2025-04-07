# [WIP] Obfuscated Adversarial Training

Fine-tune LLM to supposedly make it better for white-box monitoring. This statement is to be yet confirmed.

## Dependencies

- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- datasets
- wandb (code related to that has been removed)
- tqdm
- numpy
- scikit-learn

## Usage

```bash
python compute_oat_probes.py --masking-type instruction --probe-type linear
```

Arguments:
- `--masking-type`: Type of masking to use (`instruction` or `generation`)
- `--probe-type`: Type of probe to use (`linear` or `nonlinear`)
- `--no-lora-probes`: Disable LoRA probes (enabled by default)

## File Structure

- `compute_oat_probes.py`: Main script to run the probe training
- `src/`: Module containing core functionality
  - `__init__.py`: Imports and exposes main components
  - `probe_archs.py`: Probe architecture definitions (Linear, Nonlinear)
  - `probe_training.py`: Core training functionality
  - `encoders.py`: Wrappers for language models with SAE capabilities
  - `attacks.py`: Implementations of adversarial attacks
  - `utils.py`: Utility functions for token masking, activation extraction, etc.

## Models Supported

- Gemma 2
- LLaMA 3

## Output

The trained probes, LoRA model, and training info are saved in the `./probe_weights_comp_only` directory with a name that reflects the training configuration.

# Medical Image Segmentation with Visual Prompts

## Introduction
This project builds upon the approach introduced in [PUNet by
marcdcfischer](https://github.com/marcdcfischer/PUNet) for medical image
segmentation using visual prompting. Our method implements a two-phase training
strategy with an encoder-decoder (UNet) architecture, phase 1 is unsupervised
pretraining, and phase 2 is supervised few-shot learning with segmentation masks
as visual cues. Our method significantly improves training efficiency and
adaptability of medical image segmentation.

The prompt-based approach offers several key advantages:
- **Unsupervised learning**: The pretraining phase requires no manual annotations, allowing the model to learn from large amounts of unlabeled medical imaging data
- **Easier adaptation**: Models can quickly adapt to new medical image types or segmentation tasks by updating prompt tokens instead of retraining the entire model
- **Efficient deployment**: The large backbone model can remain on the server while only the lightweight prompt tokens need to be stored locally on client devices
- **Reduced computational costs**: Fine-tuning for new tasks requires updating only the prompt tokens, significantly reducing the computational resources needed
- **Privacy-preserving**: Enables personalization without sharing sensitive medical data back to central servers

## Quick Start
All operations (pre-training, downstream training, and testing) are executed through `main.py`.

```bash
python main.py --mode [fit|test] --training-mode [mode] --configs [path/to/config.yml]
```

The system supports two primary execution modes:
- `fit`: For training models (default)
- `test`: For evaluating trained models

## Configuration

The system uses two types of configuration:

1. **Command-line arguments** in `main.py` for:
   - Execution mode (`fit` or `test`)
   - Training mode (encoder, decoder, or both)
   - Prompting schemes
   - Model backbone type
   - Checkpoint parameters for both backbone and prompt tokens

2. **YAML configuration files** for all other settings:
   - See `../configurations/example_configs.yml` for examples
   - Command-line args override YAML settings when duplicated
   
### Main Command-Line Arguments

```
--mode                             : 'fit' for training, 'test' for evaluation [default: fit]
--training-mode                    : Training strategy [default: self_supervised_learning_all]
--configs                          : Path to YAML configuration file [default: example_configs.yml]
--backbone                         : Model architecture to use [default: swin_unetr]
--run-name                         : Custom name for the experiment

# Prompting mechanisms
--use-encoder-prompting            : Enable prompting in the encoder
--use-decoder-prompting            : Enable prompting in the decoder

# Checkpoint handling for backbone model
--load-ckpt-backbone               : Load a backbone checkpoint
--load-ckpt-backbone-path          : Path to backbone checkpoint
--save-ckpt-backbone               : Save backbone checkpoints
--save-ckpt-backbone-path          : Path to save backbone checkpoints

# Checkpoint handling for prompt tokens
--load-ckpt-prompt-tokens          : Load prompt token checkpoints
--load-ckpt-prompt-tokens-path     : Path to prompt token checkpoint
--save-ckpt-prompt-tokens          : Save prompt token checkpoints
--save-ckpt-prompt-tokens-path     : Path to save prompt token checkpoints
```

## Training Modes

The system supports different training strategies through the `--training-mode` parameter:

| Mode | Command | Description |
|------|---------|-------------|
| Phase 1 | `--training-mode self_supervised_learning_encoder` | Pre-training for encoder only |
| Phase 2 (frozen encoder) | `--training-mode self_supervised_learning_decoder` | Train decoder with frozen encoder |
| Phase 2 (trainable encoder) | `--training-mode self_supervised_learning_all` | Train both encoder and decoder (default) |
| Downstream tasks | `--training-mode downstream` | Apply model to downstream applications |

## Prompting Mechanism

This project implements two types of visual prompting approaches:

1. **Encoder Prompting**: Add prompting tokens to the encoder stage
   ```bash
   --use-encoder-prompting
   ```

2. **Decoder Prompting**: Add prompting tokens to the decoder stage
   ```bash
   --use-decoder-prompting
   ```

These prompts help guide the segmentation process by providing visual cues in the form of segmentation masks.

## Checkpoint Management

The system maintains separate checkpoints for the backbone model and prompt tokens. Checkpoint flags and paths must be set together to properly load/save model states:

### Backbone Model Checkpoints
```bash
--load-ckpt-backbone --load-ckpt-backbone-path [path/to/checkpoint]
--save-ckpt-backbone --save-ckpt-backbone-path [path/to/save/directory]
```

### Prompt Token Checkpoints
```bash
--load-ckpt-prompt-tokens --load-ckpt-prompt-tokens-path [path/to/checkpoint]
--save-ckpt-prompt-tokens --save-ckpt-prompt-tokens-path [path/to/save/directory]
```

This separation allows for flexible transfer learning scenarios, such as using a pre-trained backbone with newly initialized prompt tokens.

## Default Paths
Default locations for configurations, checkpoints, and logs are defined in `utils/configs.py`.

## Example Scripts
For complete working examples, see bash scripts in the `../example_scripts/` directory, which include:
- Pre-training pipelines
- Downstream training procedures
- Testing workflows

## System Components
- Main execution script: `main.py`
- Model selection logic: `utils/initialization.py`
- Configuration handling: `utils/configs.py`
- Training setup: `utils.setup_fitting()`
- Testing setup: `utils.setup_testing()`

## Code Structure
The main execution flow is:
1. Parse command-line arguments
2. Load and merge YAML configurations with command-line arguments
3. Set up the appropriate trainer based on mode (`fit` or `test`)
4. Execute training or testing operations
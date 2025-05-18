# TransVG: Visual Grounding for Satellite Images

This repository contains an implementation of the TransVG (Transformer-based Visual Grounding) model for satellite imagery, based on the [TransVG](https://arxiv.org/abs/2104.08541) and [TransVG++](https://arxiv.org/pdf/2206.06619v1) architectures.

## Architecture

The model follows the TransVG++ architecture with the following components:

1. **Vision Branch** (Language Conditioned Vision Transformer)
   - Patch Embedding: Transforms the input image into patch embeddings
   - Vision Encoder Layers: Standard transformer encoder layers
   - Language Conditioned Vision Encoder Layer: Vision encoder layer with cross-attention to language features

2. **Language Branch**
   - BERT-based language encoder
   - Transformer layers on top of BERT
   - Linear projection for feature dimensionality matching

3. **Prediction Head**
   - MLP-based regression head for bounding box prediction

## Directory Structure

```
visual_grounding/
├── configs/
│   └── model_config.py     # Model configuration
├── models/
│   ├── dataloader.py       # Data loader for the SatVG dataset
│   ├── language_encoder.py # Language branch of the model
│   ├── losses.py           # Loss functions (L1 and GIoU losses)
│   ├── prediction_head.py  # MLP-based prediction head
│   ├── transvg.py          # Main TransVG model
│   └── vision_encoder.py   # Vision branch of the model
├── utils/
│   ├── logger.py           # Logging utilities
│   └── metrics.py          # Evaluation metrics
├── logs/                   # Training logs
├── checkpoints/            # Model checkpoints
├── train.py                # Training script
└── README.md               # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers (Hugging Face)
- Numpy
- Pillow
- (Optional) Weights & Biases for experiment tracking

## Dataset

The model is designed to work with the DIOR-RSVG satellite visual grounding dataset. The dataset should have the following structure:

```
dior-rsvg/
├── JPEGImages/      # Image files
├── dior-train.pth   # Training data
├── dior-val.pth     # Validation data
└── dior-test.pth    # Test data
```

## Usage

### Training

```bash
python train.py --log_name transvg_experiment --epochs 100 --batch_size 32 --lr 1e-4
```

Optional arguments:
- `--config`: Path to a custom config file
- `--log_name`: Name for log files
- `--use_wandb`: Enable Weights & Biases logging
- `--epochs`: Number of epochs to train
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--resume`: Path to checkpoint to resume training from

### Evaluation

The model can be evaluated on the test set by running:

```bash
python train.py --resume path/to/checkpoint --epochs 0
```

This will load the model from the checkpoint and only run evaluation on the test set.

## Model Training Details

The model is trained with the following strategy:

1. L1 loss and GIoU loss for bounding box regression
2. AdamW optimizer with different learning rates for different components
3. Step learning rate decay
4. Model checkpointing based on validation accuracy

## Future Improvements

- Fine-tuning of the vision and language encoders
- Data augmentation techniques specific to satellite imagery
- Multi-scale feature fusion
- Attention visualization for interpretability

## References

- TransVG: End-to-End Visual Grounding with Transformers - https://arxiv.org/abs/2104.08541
- TransVG++: End-to-End Visual Grounding with Language Conditioned Vision Transformer - https://arxiv.org/pdf/2206.06619v1 

## To Train
```
cd /home/pokle/Trans-VG
python visual_grounding/train.py --log_name transvg_experiment
```

## To Eval
```
cd /home/pokle/Trans-VG
python visual_grounding/eval.py --checkpoint /path/to/checkpoint.pth
```
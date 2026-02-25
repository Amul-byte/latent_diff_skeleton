# Joint-Aware Latent Diffusion (3-Stage)

This repository implements a 3-stage Joint-Aware Latent Diffusion framework:

1. Stage 1: Skeleton latent diffusion pretraining.
2. Stage 2: IMU-to-latent regression with frozen Stage 1 encoder.
3. Stage 3: Conditional diffusion, decoding, and skeleton classification.

## Training

```bash
python train.py --stage 1
python train.py --stage 2
python train.py --stage 3
```

All stages run with synthetic toy data by default when no dataset files are provided.

## Generation

```bash
python generate.py --classify
```

This loads Stage 1 and Stage 2 checkpoints, performs conditional latent sampling, decodes skeletons, and optionally classifies generated sequences.

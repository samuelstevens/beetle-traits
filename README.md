# Beetle Traits

This repo annotates beetle elytra widths and lengths from top-down specimen images of beetles.

## Getting Started

Clone the repo.

Then run the training script:

```sh
uv run launch.py \
  --tags frozen \
  --data.hf-root /fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles \
  --data.no-include-polylines \
  --val-every 10000000 \
  --save-every 50 \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  --n-hours 1 \
  model:frozen \
  --model.dinov3-ckpt /fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx
```

## Data

https://huggingface.co/datasets/imageomics/2018-NEON-beetles

https://huggingface.co/datasets/imageomics/Hawaii-beetles

uv run src/btx/scripts/format_hawaii.py --ignore-errors --sample-rate 5 --hf-root /fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles/ --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 4

## Modeling



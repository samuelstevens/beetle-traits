# Beetle Traits

This repo annotates beetle elytra widths and lengths from top-down specimen images of beetles.

## Data

https://huggingface.co/datasets/imageomics/2018-NEON-beetles

https://huggingface.co/datasets/imageomics/Hawaii-beetles

uv run src/btx/scripts/format_hawaii.py --ignore-errors --sample-rate 5 --hf-root /fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles/ --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 4

## Modeling

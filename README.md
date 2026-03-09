# Beetle Traits

This repo annotates beetle elytra widths and lengths from top-down specimen images of beetles.

## Getting Started

Clone the repo.

Then run the training script (on the login node :smiling_imp:):

```sh
uv run launch.py \
  --tags frozen \
  --data.hf-root /fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles \
  --data.no-include-polylines \
  --val-every 10 \
  --save-every 50 \
  model:frozen \
  --model.dinov3-ckpt /fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx
```

If nothing blows up, and you see some training logs, you're training!

## Testing

`just test` runs fast unit tests. `just test-all` also runs slow integration tests (e2e train-then-infer, real ViT checkpoints). The `.env` file sets `DINOV3_JAX_CKPTS` and `DINOV3_PT_CKPTS` so that conftest can find checkpoint files; `just` loads `.env` automatically via `set dotenv-load`.

## Data

- https://huggingface.co/datasets/imageomics/2018-NEON-beetles
- https://huggingface.co/datasets/imageomics/Hawaii-beetles

```sh
uv run src/btx/scripts/format_hawaii.py --ignore-errors --sample-rate 5 --hf-root /fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles/ --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 4
```

## Workflow

### Overview:

1. Run `uv run launch.py train` with all the annotated training data.
2. With the best checkpoint(s), run `uv run launch.py inference` to get model predictions on both labeled and unlabeled data.
3. Run `uv run launch.py rank` to do k-means clustering on the unlabeled data and thenk rank within each cluster by model "confidence" (heatmap entropy).
4. Annotate the group images from the previous steps.
5. Repeat until the 95th percentile error on the validation split is below 0.3%.

There are a couple knobs to turn to get to step 5 faster:

- Better training (larger ViT, larger decoder head, better hyperparameters, training for longer, fine-tuning the ViT, etc).
- More training data (annotate more data)
- Better active learning (pick out data that is more likely to be helpful and to drive improvements in worst-case prediction error).
- Change 0.3% to a larger number (only Aly can do this).

### Details:

Here are more details for each step.
If you feel that a detail or lesson is missing, feel free to add to this list.

#### Training

Write a new experiment with a config sweep file.
Call it `train.py` just out of convention.
Run `uv run launch.py train --sweep docs/experiments/00N-NAME/sweeps/train.py` to train a keypoint predictor with all the annotated training data.
This will produce a `RUN_ID.eqx` file for each run.

#### Inference

For each `.eqx` checkpoint on disk, you can use this newly trained model to make predictions on both the labeled and unlabeled beetle images.
You need to make a new sweep file for inference (called `inference.py` out of convention).
Then you can run `uv run launch.py inference --sweep docs/experiments/00N-NAME/sweeps/inference.py`.
This will produce a `RUN_ID_labeled.parquet` and `RUN_ID_unlabeled.parquet` file with predictions, [CLS] embeddings and entropy values for both labeled and unlabeled values.

#### Ranking

With these predictions, you can cluster by the [CLS] embedding using k-means and rank within the clusters with heatmap entropy (a measure of the model's confidence).
`uv run launch.py rank --sweep docs/experiments/00N-NAME/sweeps/rank.py` for the run ids that you are using.
This will produce a ranked list of group images to annotate.

#### Annotating






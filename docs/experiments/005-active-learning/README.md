# Experiment 005: Iterative Active Learning

Goal: build the infrastructure to run an iterative annotation cycle -- train model, run inference, identify failure modes, select informative samples for human correction, retrain.

The model (exp 004, Gaussian heatmap regression) trains on Hawaii + BeetlePalooza and evaluates on BioRepo (Arizona beetles). We need to close the loop: find where the model fails, get those samples corrected, and retrain.

## Strategy

Uncertainty (heatmap entropy) + diversity (K-means on DINOv2 CLS embeddings). No sophisticated deep AL methods -- benchmarks show they frequently underperform random selection and add complexity. See `plan.md` for details and references.

## Status

Steps 1-3 are merged. Steps 4-6 are next.

1. **Checkpoint save/load** -- `btx.modeling.save_ckpt` / `load_ckpt` in `src/btx/modeling/__init__.py`. End-of-training save wired into `train.py`. Periodic/best-val saves deferred to GitHub issue #15.
2. **`extract_features`** -- `heatmap.Model.extract_features()` returns both heatmap logits and the ViT CLS token.
3. **Inference script** -- `inference.py` loads a checkpoint, runs batched inference, writes per-sample results (predictions, errors, CLS embeddings, heatmap entropy) to Parquet. Covered by e2e test in `tests/test_inference.py`.
4. **Analysis notebook** -- `notebooks/active_learning.py` (marimo). Per-species error tables, uncertainty vs error scatter, UMAP of CLS embeddings, sample selection.
5. **Annotation export** -- export images with model predictions overlaid + CSV manifest for annotators.
6. **Annotation import** -- read corrected annotations, merge into dataset.

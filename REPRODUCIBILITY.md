# Reproducibility guide

## Reference configuration

The default arguments in `project_init.py` represent the primary HRVConformer configuration in the under-review manuscript.

| Setting | Value | Interpretation |
|---|---:|---|
| Input length | 1,200 samples | Five minutes at 4 Hz |
| Patch size | 100 samples | 25 seconds |
| Embedding dimension | 144 | Patch/token width |
| Conformer blocks | 3 | Encoder depth |
| Attention heads | 8 | Relative-position multi-head attention |
| Depthwise-convolution kernel | 11 | Within each Conformer block |
| Classifier | FCN | Kernel size 11 |
| Dropout | 0.3 | Applied throughout the model |
| Optimizer | AdamW | beta1=0.85, beta2=0.998 |
| Learning rate | 6e-5 | Cosine warmup schedule |
| Warmup | 50 epochs | Minimum learning rate 1e-6 |
| Weight decay | 0.1 | AdamW regularization |
| Label smoothing | 0.2 | Cross-entropy loss |
| Batch size | 1,024 | Window-level batches |
| Training duration | 1,800 epochs | Best checkpoint selected by moving validation AUC |

The paper reports training on an NVIDIA L40S GPU under SLURM. Hardware, CUDA/cuDNN versions, and nondeterministic GPU kernels can cause small numerical differences even with fixed seeds.

## Evaluation units

The network produces two logits for every five-minute window. The repository reports both window-level and one-hour epoch-level measures:

- Window accuracy/AUC operate directly on individual windows.
- Epoch accuracy uses the most frequently predicted window class for each `file_id`.
- Epoch ROC-AUC averages positive-class probabilities across all windows with the same `file_id` before constructing the ROC curve.

Do not compare window-level and epoch-level values as though they were the same estimand.

## Experimental protocol

1. Preprocess the authorized ECG data with the enhanced Pan-Tompkins workflow.
2. Retain fixed-length NN-interval windows and one-hour epochs using the manuscript quality criteria.
3. Use ANSeR2 strong and weak labels for development.
4. Select 20% of development epochs for validation with `--seed_epoch`; train on the remainder.
5. Select checkpoints and hyperparameters using validation data only.
6. Evaluate the fixed model on the independent ANSeR1 expert-labelled cohort.
7. For matched repeated experiments, reuse the same split and initialization seeds for each baseline comparison.

The paper's aggregate table contains ten matched runs: five validation-split variations and five stochastic-initialization variations. A single invocation of `main.py` performs one training run; reproducing the full table requires external job orchestration with the corresponding split and initialization seeds.

## What is and is not reproducible here

The architecture and training/evaluation implementation are public. Exact numerical reproduction additionally requires governed access to the cohort data, the derived preprocessing artifacts, the run seeds, and equivalent compute software/hardware. Baseline models, attention-analysis notebooks, and trained checkpoints are not currently included, so the complete set of manuscript tables and figures cannot be regenerated from a clean clone alone.

This limitation is stated explicitly to distinguish code availability from full artifact reproducibility.

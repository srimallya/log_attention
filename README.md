# log_attention

`log_attention` is a compact PyTorch character-language-modeling experiment for comparing ordinary causal attention against a learned log-decayed causal attention rule.

The main script trains a small GPT-style character model on `input.txt`, optionally runs a dot-attention baseline and a log-decay model side by side, saves best checkpoints, logs metrics to CSV, generates samples during training, and creates post-training plots plus an attention-evolution video.

The repository is intentionally small:

```text
.
├── input.txt
├── train.py
├── LICENSE
└── README.md
```

## What It Does

The experiment asks a simple question:

Can a transformer keep useful long-range context while applying an explicit learned penalty to older tokens?

Standard causal attention lets every previous token compete using content similarity alone. The log-decay variant keeps that content score, but subtracts a learned distance cost that grows as the key token gets older. Nearby context starts with an advantage, but older tokens can still win if their content score is strong enough.

The script supports three modes:

| Mode | Behavior |
| --- | --- |
| `dot` | Train only the standard causal attention model. |
| `log_decay` | Train only the log-decayed attention model. |
| `all` | Train both models side by side on the same batches. |

By default, the script is set up as an experiment runner rather than a library. Most configuration lives in the `config` dictionary near the top of `train.py`.

## Quick Start

Install the runtime dependencies:

```bash
python3 -m pip install torch matplotlib
```

Then run training:

```bash
python3 train.py
```

The script automatically selects the best available device:

1. CUDA, if available
2. Apple MPS, if available
3. CPU fallback

Training reads from `input.txt`. Replace that file with any plain-text corpus you want to model.

## Configuration

Open `train.py` and edit the `config` dictionary.

Important fields:

| Field | Meaning |
| --- | --- |
| `input_path` | Path to the text corpus. |
| `val_frac` | Fraction of the corpus reserved for validation. |
| `block_size` | Context length visible to the student model. |
| `batch_size` | Number of sequences per training batch. |
| `max_steps` | Number of optimizer steps. |
| `eval_interval` | How often to evaluate, sample, log metrics, and capture attention snapshots. |
| `n_layer` | Number of transformer blocks. |
| `n_head` | Number of attention heads. |
| `n_embd` | Embedding width. |
| `attention_method` | `dot`, `log_decay`, or `all`. |
| `decay_alpha_init` | Initial learned decay strength. |
| `decay_alpha_scale` | Scale applied after `softplus` to produce alpha. |
| `use_self_distill` | Enables EMA-teacher self-distillation. |
| `enable_post_training_plots` | Saves post-training PNG plots. |
| `enable_attention_video` | Saves the side-by-side attention evolution video. |

For most first runs, keep `attention_method="all"` so the standard and log-decay models are trained under matching conditions.

## Outputs

Training writes artifacts under `checkpoints_log_decay/`.

```text
checkpoints_log_decay/
├── metrics.csv
├── dot_best.pt
├── log_decay_best.pt
├── attention_evolution.mp4
└── plots/
    ├── 01_train_val_lm_loss.png
    ├── 02_val_loss_delta.png
    ├── 03_best_val_lm.png
    ├── 04_alpha_evolution.png
    ├── 05_layerwise_alpha.png
    ├── 06_long_range_attention_mass.png
    ├── 07_mean_attention_distance.png
    ├── 08_val_loss_vs_attention_distance.png
    └── 09_best_val_lm_bar.png
```

Generated checkpoints and plots are ignored by Git so experiments do not bloat the repository.

## Plots

At the end of training, `train.py` reads `metrics.csv` and creates nine plots.

| Plot | Purpose |
| --- | --- |
| Train/validation LM loss | Shows optimization and generalization over time. |
| Validation loss delta | Shows `log_decay - dot`; values below zero favor log-decay. |
| Best validation LM loss | Shows checkpoint quality over time. |
| Alpha evolution | Shows how the learned decay strength changes. |
| Layer-wise alpha | Shows whether different layers learn different decay behavior. |
| Long-range attention mass | Tracks attention assigned to older tokens. |
| Mean attention distance | Tracks the average attended distance. |
| Val loss vs attention distance | Relates attention behavior to validation quality. |
| Best validation bar chart | Summarizes final best validation loss by model. |

## Attention Evolution Video

When `enable_attention_video=True`, the script captures a fixed validation probe at every eval interval. For each model and each layer, it averages attention over batch and heads and stores a layer-level attention matrix.

After training, those snapshots become:

```text
checkpoints_log_decay/attention_evolution.mp4
```

The video layout is:

- rows: transformer layers
- columns: trained attention methods, such as `dot` and `log_decay`
- frames: evaluation steps

This makes it easier to inspect whether the log-decay model merely changes loss, or actually learns a different attention geometry over time.

If `ffmpeg` is not available, the script falls back to a GIF.

## Checkpoints

Whenever validation loss improves, the script saves a best checkpoint for that model:

```text
checkpoints_log_decay/dot_best.pt
checkpoints_log_decay/log_decay_best.pt
```

Each checkpoint contains:

- model weights
- config
- vocabulary size
- character-to-index mapping
- index-to-character mapping
- step number
- best validation statistics

## Self-Distillation

The script includes an optional self-distillation path controlled by:

```python
use_self_distill=True
```

In this mode:

1. The student sees a normal context window.
2. The EMA teacher sees the same context plus a privileged prefix.
3. The student optimizes language-modeling cross entropy plus a KL distillation term.
4. Distillation can be entropy-filtered so the student only learns from teacher positions where the teacher is more confident.

This is off by default. Enable it when you specifically want to test whether a teacher with extra prefix context can transfer useful information into the normal-context student.

## Practical Notes

This is a character-level model, so it can train on arbitrary text without a tokenizer. That keeps the experiment easy to run and inspect, but it also means losses are character-level losses, not token-level losses from a modern subword tokenizer.

The default model is intentionally small enough for local experimentation, but side-by-side training with `attention_method="all"` doubles the model work. If training is slow, reduce:

- `n_embd`
- `n_layer`
- `n_head`
- `block_size`
- `batch_size`
- `max_steps`

For faster artifact generation while testing the plotting pipeline, lower `max_steps` and `eval_interval`.

## Reproducibility

The script sets:

```python
torch.manual_seed(config["seed"])
```

This improves repeatability, but exact results can still differ by device, PyTorch backend, and nondeterministic kernels.

## License

This repository is licensed under the MIT License. See `LICENSE`.

## Math Details

Let a sequence have positions `0, 1, ..., T - 1`. For a query at position `i` and a key at position `j`, causal attention only allows `j <= i`.

For each head, standard scaled dot-product attention computes:

```text
s_ij = (q_i . k_j) / sqrt(d_h)
```

where:

- `q_i` is the query vector at position `i`
- `k_j` is the key vector at position `j`
- `d_h` is the per-head dimensionality

Causal masking sets scores with `j > i` to `-inf`, then attention weights are:

```text
a_ij = exp(s_ij) / sum_{m <= i} exp(s_im)
```

The log-decayed variant modifies the score before the softmax:

```text
s_ij = (q_i . k_j) / sqrt(d_h) - alpha_h log(1 + i - j)
```

where:

- `i - j` is the causal distance from query to key
- `log(1 + i - j)` is zero for the current token and grows slowly for older tokens
- `alpha_h >= 0` is learned per head

The implementation parameterizes alpha as:

```text
alpha_h = softplus(raw_alpha_h) * decay_alpha_scale
```

This keeps alpha non-negative while allowing unconstrained optimization of `raw_alpha_h`.

The resulting attention weights are:

```text
a_ij =
    exp((q_i . k_j) / sqrt(d_h) - alpha_h log(1 + i - j))
    -------------------------------------------------------
    sum_{m <= i} exp((q_i . k_m) / sqrt(d_h) - alpha_h log(1 + i - m))
```

Using the identity `exp(-alpha log(x)) = x^{-alpha}`, this can be rewritten as:

```text
a_ij =
    exp((q_i . k_j) / sqrt(d_h)) * (1 + i - j)^(-alpha_h)
    ----------------------------------------------------------------
    sum_{m <= i} exp((q_i . k_m) / sqrt(d_h)) * (1 + i - m)^(-alpha_h)
```

So the log-decay term is equivalent to multiplying the usual content-based attention score by a power-law distance prior before normalization.

Interpretation:

- `alpha_h = 0` recovers standard causal attention.
- Small `alpha_h` mildly prefers recent context.
- Large `alpha_h` strongly penalizes older context.
- Older tokens can still receive high attention if their content score is high enough.

This is different from a hard local window. A hard window forbids distant tokens. Log decay keeps distant tokens available but makes them pay a learned distance tax.

## Diagnostics

The script logs several attention diagnostics.

Mean attention distance:

```text
D_mean = mean_{batch, head, i} sum_{j <= i} a_ij (i - j)
```

This estimates how far back the model attends on average.

Long-range attention mass:

```text
M_long = mean_{batch, head, i} sum_{j <= i, i - j >= tau} a_ij
```

where:

```text
tau = floor(block_size * long_range_fraction)
```

This tracks how much probability mass goes to older positions.

For log-decay models, the script also logs:

```text
alpha_mean
alpha_min
alpha_max
L0_alpha_mean
L1_alpha_mean
...
```

Layer-wise alpha is useful because early layers may learn local character composition while later layers may preserve broader context.

## Distillation Objective

When self-distillation is enabled, the student loss is:

```text
L = L_CE + lambda_KD L_KD
```

The cross-entropy term is ordinary next-character prediction:

```text
L_CE = - mean log p_student(y_t | x_<=t)
```

The distillation term compares softened teacher and student distributions:

```text
L_KD = T^2 KL(
    softmax(z_teacher / T)
    ||
    softmax(z_student / T)
)
```

where:

- `T` is `distill_temperature`
- `z_teacher` is the EMA teacher logits
- `z_student` is the student logits
- `lambda_KD` is `distill_weight`

If entropy filtering is enabled, the KL term is only applied where the teacher is sufficiently more confident than the student:

```text
H_teacher < H_student - teacher_entropy_margin
```

with entropy:

```text
H(p) = - sum_c p_c log p_c
```

This avoids forcing the student to imitate uncertain teacher predictions.

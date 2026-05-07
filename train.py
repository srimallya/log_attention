# ============================
# Log-Decayed Causal Attention Character LM + Self-Distillation
# Corpus: input.txt
#
# Core idea:
#   Standard attention:
#       score(i, j) = Q_i K_j / sqrt(d)
#
#   Log-decayed attention:
#       score(i, j) = Q_i K_j / sqrt(d) - alpha_h * log(1 + i - j)
#
#   Meaning:
#       nearby context is sharp by default
#       older context fades by design
#       old tokens can still win if semantic/content score is strong
#
# Supports attention_method:
#   "dot"        -> standard causal attention
#   "log_decay"  -> causal attention with learned per-head logarithmic distance tax
#   "all"        -> train dot and log_decay side-by-side
#
# Adds:
#   - EMA teacher
#   - privileged-prefix teacher context
#   - CE + KL self-distillation
#   - entropy-filtered KD
#   - learned per-head decay alpha
#   - diagnostics for alpha and long-range attention mass
#
# No semantic-window cosplay. Just time with friction.
# ============================

import math
import os
import copy
import csv
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# Hyperparameters
# ============================

config = dict(
    # data
    input_path="/Users/srimallyamaitra/codes/log_attention/input.txt",
    val_frac=0.10,

    # sequence
    block_size=256,
    hint_len=64,

    # training
    batch_size=8,
    max_steps=25000,
    eval_interval=200,
    eval_iters=10,
    learning_rate=3e-4,
    weight_decay=0.01,
    grad_clip=1.0,

    # model
    n_layer=4,
    n_head=8,
    n_embd=512,
    dropout=0.20,

    # attention
    attention_method="all",  # "dot", "log_decay", "all"

    # log decay attention
    # alpha is learned per layer/head:
    #   alpha = softplus(raw_alpha) * decay_alpha_scale
    #
    # Bigger alpha = older tokens fade harder.
    # Smaller alpha = closer to ordinary attention.
    decay_alpha_init=0.10,
    decay_alpha_scale=0.5,

    # Optional lower/upper clamp for diagnostics only.
    # The actual alpha is not hard-clamped unless use_alpha_clamp=True.
    use_alpha_clamp=False,
    alpha_min=0.0,
    alpha_max=8.0,

    # Attention diagnostics
    long_range_fraction=0.50,  # report attention mass going to tokens older than 50% of context

    # persistence
    ckpt_dir="checkpoints_log_decay",
    csv_log_path="checkpoints_log_decay/metrics.csv",
    plot_dir="checkpoints_log_decay/plots",
    attention_video_path="checkpoints_log_decay/attention_evolution.mp4",
    enable_post_training_plots=True,
    enable_attention_video=True,
    attention_video_fps=2,
    attention_video_probe_batch=1,

    # self-distillation
    use_self_distill=False,
    distill_weight=0.15,
    distill_temperature=2.0,
    ema_decay=0.995,
    entropy_filter=True,
    teacher_entropy_margin=0.05,

    # sampling
    sample_tokens=300,
    top_k=90,
    temperature=0.8,
    sample_prefix="",

    # reproducibility
    seed=1337,
)


# ============================
# Setup
# ============================

torch.manual_seed(config["seed"])


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()
print(f"using device: {device}")

path = config["input_path"]
if not os.path.exists(path):
    raise FileNotFoundError(
        f"input.txt not found at {path}. Set config['input_path'] correctly. "
        "The machine cannot train on vibes. Tragic, really."
    )

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

if len(text) < config["block_size"] + config["hint_len"] + 2:
    raise ValueError(
        "Corpus is too small for block_size + hint_len. "
        "Feed the tiny beast more text."
    )

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)


def decode(t):
    return "".join([itos[int(i)] for i in t])


data = encode(text)

n = int((1.0 - config["val_frac"]) * len(data))
train_data = data[:n]
val_data = data[n:]

if len(val_data) < config["block_size"] + config["hint_len"] + 2:
    print(
        "warning: validation split is tiny. "
        "Eval may be noisy, because apparently statistics also requires food."
    )


# ============================
# Data
# ============================

def get_batch(split):
    src = train_data if split == "train" else val_data

    if len(src) <= config["block_size"] + 1:
        raise ValueError(f"{split} split too small for block_size.")

    ix = torch.randint(
        0,
        len(src) - config["block_size"] - 1,
        (config["batch_size"],),
    )

    x = torch.stack([
        src[i:i + config["block_size"]]
        for i in ix
    ]).to(device)

    y = torch.stack([
        src[i + 1:i + 1 + config["block_size"]]
        for i in ix
    ]).to(device)

    return x, y


def get_prefix_distill_batch(split, context_len, hint_len):
    src = train_data if split == "train" else val_data
    total = hint_len + context_len + 1

    if len(src) <= total:
        raise ValueError(
            f"{split} split too small for hint_len + block_size. "
            "The teacher cannot see privileged context if there is no context. Cruel, but mathematical."
        )

    ix = torch.randint(0, len(src) - total, (config["batch_size"],))
    seq = torch.stack([src[i:i + total] for i in ix]).to(device)

    x_student = seq[:, hint_len:hint_len + context_len]
    y_student = seq[:, hint_len + 1:hint_len + context_len + 1]

    # Teacher sees extra prefix before the student's visible context.
    x_teacher = seq[:, :hint_len + context_len]

    return x_student, y_student, x_teacher


# ============================
# Utility
# ============================

def top_k_filter(logits, k):
    if k is None or k <= 0:
        return logits

    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    cutoff = v[..., -1, None]

    return torch.where(
        logits < cutoff,
        torch.full_like(logits, -1e10),
        logits,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================
# Attention
# ============================

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        block_size,
        attention_method,
        decay_alpha_init,
        decay_alpha_scale,
        use_alpha_clamp,
        alpha_min,
        alpha_max,
        long_range_fraction,
    ):
        super().__init__()

        assert n_embd % n_head == 0
        assert attention_method in ["dot", "log_decay"]

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size
        self.attention_method = attention_method

        self.decay_alpha_scale = decay_alpha_scale
        self.use_alpha_clamp = use_alpha_clamp
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.long_range_fraction = long_range_fraction

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        # dist[i, j] = i - j for causal past tokens.
        pos = torch.arange(block_size)
        dist = pos[:, None] - pos[None, :]
        dist = dist.clamp(min=0).float()

        self.register_buffer(
            "log_distance",
            torch.log1p(dist).view(1, 1, block_size, block_size),
        )

        # We parameterize alpha through inverse softplus approximately.
        # alpha = softplus(raw_alpha) * scale
        init = float(decay_alpha_init) / max(float(decay_alpha_scale), 1e-8)
        raw_init = math.log(math.exp(init) - 1.0) if init > 1e-6 else -10.0

        self.raw_alpha = nn.Parameter(torch.full((n_head,), raw_init))

        self.last_diag = {}
        self.capture_attention_map = False
        self.last_attention_map = None

    def get_alpha(self):
        alpha = F.softplus(self.raw_alpha) * self.decay_alpha_scale

        if self.use_alpha_clamp:
            alpha = alpha.clamp(self.alpha_min, self.alpha_max)

        return alpha

    def split_heads(self, x):
        B, T, C = x.size()
        return x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    def merge_heads(self, y):
        B, H, T, D = y.size()
        return y.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, x):
        B, T, C = x.size()

        k = self.split_heads(self.key(x))
        q = self.split_heads(self.query(x))
        v = self.split_heads(self.value(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.attention_method == "log_decay":
            alpha = self.get_alpha().view(1, self.n_head, 1, 1)
            decay_bias = -alpha * self.log_distance[:, :, :T, :T]
            scores = scores + decay_bias
        else:
            alpha = None

        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf"),
        )

        att = F.softmax(scores, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = self.merge_heads(y)
        y = self.resid_drop(self.proj(y))

        with torch.no_grad():
            diag = {}

            if self.attention_method == "log_decay":
                a = self.get_alpha()
                diag["alpha_mean"] = float(a.mean().detach().cpu())
                diag["alpha_min"] = float(a.min().detach().cpu())
                diag["alpha_max"] = float(a.max().detach().cpu())

            # Long-range attention mass diagnostic.
            # Reports average attention paid to tokens older than fraction of current block.
            dist = torch.arange(T, device=x.device)[:, None] - torch.arange(T, device=x.device)[None, :]
            dist = dist.clamp(min=0)

            threshold = max(1, int(T * self.long_range_fraction))
            long_mask = (dist >= threshold).view(1, 1, T, T)

            causal_count = long_mask.float().sum().clamp_min(1.0)
            long_mass = (att * long_mask.float()).sum() / (B * self.n_head * T)

            diag["long_attn_mass"] = float(long_mass.detach().cpu())

            # Mean attention distance.
            # Useful to see whether the model actually shifts local/global behavior.
            dist_f = dist.float().view(1, 1, T, T)
            mean_dist = (att * dist_f).sum(dim=-1).mean()
            diag["mean_attn_dist"] = float(mean_dist.detach().cpu())

            self.last_diag = diag

            if self.capture_attention_map:
                # Mean over batch and heads gives one stable layer-level attention image.
                self.last_attention_map = att.mean(dim=(0, 1)).detach().cpu()

        return y


# ============================
# Transformer Block
# ============================

class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        block_size,
        attention_method,
        decay_alpha_init,
        decay_alpha_scale,
        use_alpha_clamp,
        alpha_min,
        alpha_max,
        long_range_fraction,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            block_size=block_size,
            attention_method=attention_method,
            decay_alpha_init=decay_alpha_init,
            decay_alpha_scale=decay_alpha_scale,
            use_alpha_clamp=use_alpha_clamp,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            long_range_fraction=long_range_fraction,
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================
# GPT Character LM
# ============================

class GPTLogDecayLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layer,
        n_head,
        n_embd,
        dropout,
        block_size,
        attention_method,
        decay_alpha_init,
        decay_alpha_scale,
        use_alpha_clamp,
        alpha_min,
        alpha_max,
        long_range_fraction,
    ):
        super().__init__()

        assert attention_method in ["dot", "log_decay"]

        self.block_size = block_size
        self.attention_method = attention_method

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            h=nn.ModuleList([
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    block_size=block_size,
                    attention_method=attention_method,
                    decay_alpha_init=decay_alpha_init,
                    decay_alpha_scale=decay_alpha_scale,
                    use_alpha_clamp=use_alpha_clamp,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    long_range_fraction=long_range_fraction,
                )
                for _ in range(n_layer)
            ]),
            ln_f=nn.LayerNorm(n_embd),
        ))

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying.
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.block_size, f"T={T} exceeds block_size={self.block_size}"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss_lm = None

        if targets is not None:
            loss_lm = F.cross_entropy(
                logits.reshape(B * T, -1),
                targets.reshape(B * T),
            )

        return logits, loss_lm

    def get_attention_diagnostics(self):
        values = {}
        counts = {}

        for layer_idx, block in enumerate(self.transformer.h):
            diag = block.attn.last_diag

            for key, val in diag.items():
                values[key] = values.get(key, 0.0) + float(val)
                counts[key] = counts.get(key, 0) + 1

                layer_key = f"L{layer_idx}_{key}"
                values[layer_key] = float(val)
                counts[layer_key] = 1

        return {
            key: values[key] / max(counts[key], 1)
            for key in values
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k_val=0):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond, targets=None)

            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = top_k_filter(logits, top_k_val)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ============================
# Model Builders
# ============================

AVAILABLE_METHODS = ["dot", "log_decay"]


def model_block_size():
    if config["use_self_distill"]:
        return config["block_size"] + config["hint_len"]
    return config["block_size"]


def make_model(attention_method):
    return GPTLogDecayLM(
        vocab_size=vocab_size,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        block_size=model_block_size(),
        attention_method=attention_method,
        decay_alpha_init=config["decay_alpha_init"],
        decay_alpha_scale=config["decay_alpha_scale"],
        use_alpha_clamp=config["use_alpha_clamp"],
        alpha_min=config["alpha_min"],
        alpha_max=config["alpha_max"],
        long_range_fraction=config["long_range_fraction"],
    ).to(device)


def build_experiment_models():
    method = config["attention_method"]

    if method == "all":
        names = AVAILABLE_METHODS
    elif method in AVAILABLE_METHODS:
        names = [method]
    else:
        raise ValueError(f"Unknown attention_method: {method}")

    models = {
        name: make_model(name)
        for name in names
    }

    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        for name, model in models.items()
    }

    return models, optimizers


def build_ema_teachers(models):
    teachers = {}

    for name, model in models.items():
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()

        for p in teacher.parameters():
            p.requires_grad_(False)

        teachers[name] = teacher

    return teachers


@torch.no_grad()
def update_ema_teacher(teacher, student, decay):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


# ============================
# Self-Distillation
# ============================

def token_entropy_from_logits(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def distill_kl_loss(
    student_logits,
    teacher_logits,
    temperature=2.0,
    entropy_filter=True,
    teacher_entropy_margin=0.05,
):
    T = temperature

    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)

    token_kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)

    with torch.no_grad():
        teacher_entropy = token_entropy_from_logits(
            teacher_logits,
            temperature=T,
        )

        student_entropy = token_entropy_from_logits(
            student_logits,
            temperature=T,
        )

        if entropy_filter:
            # Keep KD where teacher is more confident than student.
            mask = teacher_entropy < (student_entropy - teacher_entropy_margin)
        else:
            mask = torch.ones_like(token_kl, dtype=torch.bool)

        mask_f = mask.float()
        denom = mask_f.sum().clamp_min(1.0)

    kd = (token_kl * mask_f).sum() / denom
    kd = kd * (T * T)

    stats = {
        "teacher_entropy": float(teacher_entropy.mean().detach().cpu()),
        "student_entropy": float(student_entropy.mean().detach().cpu()),
        "kd_keep": float(mask_f.mean().detach().cpu()),
    }

    return kd, stats


# ============================
# Eval / Sampling
# ============================

@torch.no_grad()
def estimate_loss(model):
    model.eval()

    out = {}

    for split in ["train", "val"]:
        losses_lm = []

        for _ in range(config["eval_iters"]):
            xb, yb = get_batch(split)
            _, loss_lm = model(xb, yb)
            losses_lm.append(loss_lm.item())

        out[split] = dict(
            lm=sum(losses_lm) / len(losses_lm),
        )

    model.train()
    return out


@torch.no_grad()
def sample_text(model, prefix=None, steps=None):
    was_training = model.training
    if prefix is None:
        prefix = config["sample_prefix"]

    if steps is None:
        steps = config["sample_tokens"]

    if len(prefix) == 0:
        start_id = torch.randint(0, vocab_size, (1, 1), device=device)
    else:
        start_id = encode(prefix).unsqueeze(0).to(device)

    out = model.generate(
        start_id,
        max_new_tokens=steps,
        temperature=config["temperature"],
        top_k_val=config["top_k"],
    )[0].tolist()

    print(decode(out))

    if was_training:
        model.train()


# ============================
# Training Steps
# ============================

def train_one_model_step(model, optimizer, xb, yb):
    _, loss_lm = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss_lm.backward()

    grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        config["grad_clip"],
    )

    optimizer.step()

    diag = model.get_attention_diagnostics()

    return dict(
        loss=loss_lm.item(),
        lm=loss_lm.item(),
        grad=float(grad_norm),
        diag=diag,
    )


def train_one_model_step_self_distill(model, teacher, optimizer):
    context_len = config["block_size"]
    hint_len = config["hint_len"]

    x_student, y_student, x_teacher = get_prefix_distill_batch(
        split="train",
        context_len=context_len,
        hint_len=hint_len,
    )

    student_logits, ce_loss = model(x_student, y_student)

    with torch.no_grad():
        teacher.eval()

        teacher_logits_full, _ = teacher(
            x_teacher,
            targets=None,
        )

        # Align teacher's predictions to student's positions.
        teacher_logits = teacher_logits_full[
            :,
            hint_len:hint_len + context_len,
            :
        ]

    kd_loss, kd_stats = distill_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=config["distill_temperature"],
        entropy_filter=config["entropy_filter"],
        teacher_entropy_margin=config["teacher_entropy_margin"],
    )

    loss = ce_loss + config["distill_weight"] * kd_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        config["grad_clip"],
    )

    optimizer.step()

    update_ema_teacher(
        teacher=teacher,
        student=model,
        decay=config["ema_decay"],
    )

    diag = model.get_attention_diagnostics()

    return dict(
        loss=loss.item(),
        ce=ce_loss.item(),
        kd=kd_loss.item(),
        grad=float(grad_norm),
        diag=diag,
        kd_stats=kd_stats,
    )


# ============================
# Logging
# ============================

def format_diag(diag):
    if not diag:
        return ""

    pieces = []

    for key in sorted(diag.keys()):
        pieces.append(f"{key}={diag[key]:.3f}")

    return " ".join(pieces)


def format_kd_stats(kd_stats):
    if not kd_stats:
        return ""

    return (
        f"Ht={kd_stats['teacher_entropy']:.3f} "
        f"Hs={kd_stats['student_entropy']:.3f} "
        f"keep={kd_stats['kd_keep']:.2f}"
    )


def format_train_cell(log):
    if log is None:
        return "".ljust(120)

    if "ce" in log:
        base = (
            f"loss={log['loss']:.4f} "
            f"ce={log['ce']:.4f} "
            f"kd={log['kd']:.4f} "
            f"grad={log['grad']:.2f}"
        )

        kd_stats = format_kd_stats(log.get("kd_stats", {}))

        if kd_stats:
            base += " | " + kd_stats

    else:
        base = (
            f"loss={log['loss']:.4f} "
            f"lm={log['lm']:.4f} "
            f"grad={log['grad']:.2f}"
        )

    diag = format_diag(log.get("diag", {}))

    if diag:
        base += " | " + diag

    return base.ljust(120)


def print_side_by_side_train(step, step_logs, models):
    cells = []

    for name in models.keys():
        cells.append(
            f"{name.upper()}: {format_train_cell(step_logs.get(name))}"
        )

    print(f"[step {step:5d}] " + " | ".join(cells))


def print_side_by_side_eval(step, stats_by_model, best_stats, models):
    print(f"\n===== eval step {step} =====")
    print("model       | train_lm | val_lm | best_val | best_step")
    print("------------|----------|--------|----------|----------")

    for name in models.keys():
        s = stats_by_model[name]
        best = best_stats[name]

        print(
            f"{name.upper():<11} | "
            f"{s['train']['lm']:.4f}   | "
            f"{s['val']['lm']:.4f} | "
            f"{best['val_lm']:.4f}   | "
            f"{best['step']:>8d}"
        )

    print("")


def checkpoint_payload(model, step, best):
    return {
        "model": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "step": step,
        "best_stats": best,
    }


def save_best_checkpoint(name, model, step, best):
    os.makedirs(config["ckpt_dir"], exist_ok=True)
    ckpt_path = os.path.join(config["ckpt_dir"], f"{name}_best.pt")

    torch.save(
        checkpoint_payload(
            model=model,
            step=step,
            best=best,
        ),
        ckpt_path,
    )

    print(f"saved best checkpoint: {ckpt_path}")


def csv_diag_fields():
    fields = [
        "alpha_mean",
        "alpha_min",
        "alpha_max",
        "mean_attn_dist",
        "long_attn_mass",
    ]

    for layer_idx in range(config["n_layer"]):
        fields.extend([
            f"L{layer_idx}_alpha_mean",
            f"L{layer_idx}_mean_attn_dist",
            f"L{layer_idx}_long_attn_mass",
        ])

    return fields


def csv_fieldnames():
    return [
        "step",
        "model",
        "train_lm",
        "val_lm",
        "best_val_lm",
    ] + csv_diag_fields()


def init_csv_logger():
    csv_dir = os.path.dirname(config["csv_log_path"])

    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(config["csv_log_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writeheader()


def append_csv_row(step, name, stats, best, diag):
    row = {
        "step": step,
        "model": name,
        "train_lm": stats["train"]["lm"],
        "val_lm": stats["val"]["lm"],
        "best_val_lm": best["val_lm"],
    }

    for key in csv_diag_fields():
        row[key] = diag.get(key, "")

    with open(config["csv_log_path"], "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writerow(row)


def print_samples(step, models):
    print("----- samples -----")

    for name, model in models.items():
        print(f"\n[{name.upper()} sample @ step {step}]")
        sample_text(
            model,
            prefix=config["sample_prefix"],
            steps=config["sample_tokens"],
        )

    print("-------------------\n")


def print_config_summary(models):
    print("\n===== config =====")
    print(f"input_path: {config['input_path']}")
    print(f"vocab_size: {vocab_size}")
    print(f"train_tokens: {len(train_data):,}")
    print(f"val_tokens: {len(val_data):,}")
    print(f"attention_method: {config['attention_method']}")
    print(f"block_size: {config['block_size']}")
    print(f"hint_len: {config['hint_len']}")
    print(f"model_block_size: {model_block_size()}")
    print(f"batch_size: {config['batch_size']}")
    print(f"n_layer: {config['n_layer']}")
    print(f"n_head: {config['n_head']}")
    print(f"n_embd: {config['n_embd']}")
    print(f"dropout: {config['dropout']}")
    print(f"learning_rate: {config['learning_rate']}")
    print(f"weight_decay: {config['weight_decay']}")
    print(f"grad_clip: {config['grad_clip']}")

    print("\n===== log decay =====")
    print(f"decay_alpha_init: {config['decay_alpha_init']}")
    print(f"decay_alpha_scale: {config['decay_alpha_scale']}")
    print(f"use_alpha_clamp: {config['use_alpha_clamp']}")
    print(f"alpha_min: {config['alpha_min']}")
    print(f"alpha_max: {config['alpha_max']}")
    print(f"long_range_fraction: {config['long_range_fraction']}")

    print("\n===== persistence =====")
    print(f"ckpt_dir: {config['ckpt_dir']}")
    print(f"csv_log_path: {config['csv_log_path']}")
    print(f"plot_dir: {config['plot_dir']}")
    print(f"attention_video_path: {config['attention_video_path']}")
    print(f"enable_post_training_plots: {config['enable_post_training_plots']}")
    print(f"enable_attention_video: {config['enable_attention_video']}")

    print("\n===== self distill =====")
    print(f"use_self_distill: {config['use_self_distill']}")
    print(f"distill_weight: {config['distill_weight']}")
    print(f"distill_temperature: {config['distill_temperature']}")
    print(f"ema_decay: {config['ema_decay']}")
    print(f"entropy_filter: {config['entropy_filter']}")
    print(f"teacher_entropy_margin: {config['teacher_entropy_margin']}")

    print("\n===== models =====")

    for name, model in models.items():
        print(f"{name.upper():<11} params: {count_parameters(model):,}")

    print("")


# ============================
# Post-Training Artifacts
# ============================

def parse_float(value):
    if value is None or value == "":
        return None

    return float(value)


def load_metric_rows():
    with open(config["csv_log_path"], "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []

        for row in reader:
            rows.append({
                key: int(value) if key == "step" else parse_float(value)
                if key != "model" else value
                for key, value in row.items()
            })

    return rows


def rows_for_model(rows, model_name):
    return sorted(
        [row for row in rows if row["model"] == model_name],
        key=lambda row: row["step"],
    )


def series(model_rows, field):
    xs = []
    ys = []

    for row in model_rows:
        value = row.get(field)

        if value is not None:
            xs.append(row["step"])
            ys.append(value)

    return xs, ys


def save_plot(fig, filename):
    os.makedirs(config["plot_dir"], exist_ok=True)
    path = os.path.join(config["plot_dir"], filename)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path


def plot_training_artifacts():
    if not config["enable_post_training_plots"]:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is not installed; skipping post-training plots.")
        return []

    rows = load_metric_rows()

    if not rows:
        print("warning: no metric rows found; skipping post-training plots.")
        return []

    model_names = sorted(set(row["model"] for row in rows))
    by_model = {name: rows_for_model(rows, name) for name in model_names}
    written = []

    # 1. Train/validation language-modeling loss.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "train_lm")
        ax.plot(xs, ys, linestyle="--", label=f"{name} train")
        xs, ys = series(model_rows, "val_lm")
        ax.plot(xs, ys, label=f"{name} val")
    ax.set_title("Train and validation LM loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "01_train_val_lm_loss.png"))
    plt.close(fig)

    # 2. Validation loss delta: log_decay - dot.
    if "dot" in by_model and "log_decay" in by_model:
        dot_by_step = {row["step"]: row["val_lm"] for row in by_model["dot"]}
        log_by_step = {row["step"]: row["val_lm"] for row in by_model["log_decay"]}
        common_steps = sorted(set(dot_by_step) & set(log_by_step))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.plot(
            common_steps,
            [log_by_step[step] - dot_by_step[step] for step in common_steps],
            label="log_decay - dot",
        )
        ax.set_title("Validation loss delta")
        ax.set_xlabel("step")
        ax.set_ylabel("delta loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        written.append(save_plot(fig, "02_val_loss_delta.png"))
        plt.close(fig)

    # 3. Best validation loss over time.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "best_val_lm")
        ax.plot(xs, ys, label=name)
    ax.set_title("Best validation LM loss")
    ax.set_xlabel("step")
    ax.set_ylabel("best val loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "03_best_val_lm.png"))
    plt.close(fig)

    # 4. Alpha mean with min/max band.
    if "log_decay" in by_model:
        log_rows = by_model["log_decay"]
        xs, alpha_mean = series(log_rows, "alpha_mean")
        _, alpha_min = series(log_rows, "alpha_min")
        _, alpha_max = series(log_rows, "alpha_max")

        if xs and alpha_mean:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(xs, alpha_mean, label="alpha mean")

            if len(alpha_min) == len(xs) and len(alpha_max) == len(xs):
                ax.fill_between(xs, alpha_min, alpha_max, alpha=0.20, label="min/max")

            ax.set_title("Log-decay alpha evolution")
            ax.set_xlabel("step")
            ax.set_ylabel("alpha")
            ax.grid(True, alpha=0.25)
            ax.legend()
            written.append(save_plot(fig, "04_alpha_evolution.png"))
            plt.close(fig)

        # 5. Layer-wise alpha.
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for layer_idx in range(config["n_layer"]):
            xs, ys = series(log_rows, f"L{layer_idx}_alpha_mean")
            if xs and ys:
                plotted = True
                ax.plot(xs, ys, label=f"L{layer_idx}")

        if plotted:
            ax.set_title("Layer-wise log-decay alpha")
            ax.set_xlabel("step")
            ax.set_ylabel("alpha")
            ax.grid(True, alpha=0.25)
            ax.legend(ncol=2)
            written.append(save_plot(fig, "05_layerwise_alpha.png"))
        plt.close(fig)

    # 6. Long-range attention mass.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "long_attn_mass")
        if xs and ys:
            ax.plot(xs, ys, label=name)
    ax.set_title("Long-range attention mass")
    ax.set_xlabel("step")
    ax.set_ylabel("attention mass")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "06_long_range_attention_mass.png"))
    plt.close(fig)

    # 7. Mean attention distance.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "mean_attn_dist")
        if xs and ys:
            ax.plot(xs, ys, label=name)
    ax.set_title("Mean attention distance")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "07_mean_attention_distance.png"))
    plt.close(fig)

    # 8. Validation loss vs attention behavior.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model_rows in by_model.items():
        xs = []
        ys = []
        for row in model_rows:
            if row.get("mean_attn_dist") is not None and row.get("val_lm") is not None:
                xs.append(row["mean_attn_dist"])
                ys.append(row["val_lm"])
        if xs and ys:
            ax.scatter(xs, ys, s=24, alpha=0.75, label=name)
    ax.set_title("Validation loss vs mean attention distance")
    ax.set_xlabel("mean attention distance")
    ax.set_ylabel("val LM loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "08_val_loss_vs_attention_distance.png"))
    plt.close(fig)

    # 9. Final best validation comparison.
    labels = []
    values = []
    for name, model_rows in by_model.items():
        best_values = [row["best_val_lm"] for row in model_rows if row.get("best_val_lm") is not None]
        if best_values:
            labels.append(name)
            values.append(min(best_values))

    if labels and values:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, values)
        ax.set_title("Best validation LM loss by model")
        ax.set_xlabel("model")
        ax.set_ylabel("best val LM loss")
        ax.grid(True, axis="y", alpha=0.25)
        written.append(save_plot(fig, "09_best_val_lm_bar.png"))
        plt.close(fig)

    print(f"saved {len(written)} plot artifacts to {config['plot_dir']}")
    return written


def set_attention_capture(model, enabled):
    for block in model.transformer.h:
        block.attn.capture_attention_map = enabled

        if enabled:
            block.attn.last_attention_map = None


@torch.no_grad()
def collect_attention_snapshot(step, models, probe_x):
    snapshot = {
        "step": step,
        "models": {},
    }

    for name, model in models.items():
        was_training = model.training
        model.eval()

        set_attention_capture(model, True)
        model(probe_x, targets=None)
        set_attention_capture(model, False)

        layers = []
        for block in model.transformer.h:
            attention_map = block.attn.last_attention_map

            if attention_map is not None:
                layers.append(attention_map.clone())

        snapshot["models"][name] = layers

        if was_training:
            model.train()

    return snapshot


def build_attention_video(attention_snapshots, model_names):
    if not config["enable_attention_video"]:
        return None

    if not attention_snapshots:
        print("warning: no attention snapshots captured; skipping attention video.")
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        print("warning: matplotlib is not installed; skipping attention video.")
        return None

    os.makedirs(os.path.dirname(config["attention_video_path"]), exist_ok=True)

    rows = config["n_layer"]
    cols = len(model_names)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.2 * cols, 3.6 * rows),
        squeeze=False,
    )

    images = []
    vmax = 0.0

    for snapshot in attention_snapshots:
        for name in model_names:
            for layer_map in snapshot["models"].get(name, []):
                vmax = max(vmax, float(layer_map.max()))

    vmax = max(vmax, 1e-6)

    for layer_idx in range(rows):
        image_row = []
        for col_idx, name in enumerate(model_names):
            ax = axes[layer_idx][col_idx]
            first_map = attention_snapshots[0]["models"][name][layer_idx].numpy()
            image = ax.imshow(
                first_map,
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
                interpolation="nearest",
                animated=True,
            )
            ax.set_title(f"{name} L{layer_idx}")
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")
            image_row.append(image)
        images.append(image_row)

    title = fig.suptitle("")
    fig.tight_layout()

    def update(frame_idx):
        snapshot = attention_snapshots[frame_idx]
        title.set_text(f"attention evolution - step {snapshot['step']}")

        artists = [title]
        for layer_idx in range(rows):
            for col_idx, name in enumerate(model_names):
                images[layer_idx][col_idx].set_array(
                    snapshot["models"][name][layer_idx].numpy()
                )
                artists.append(images[layer_idx][col_idx])

        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(attention_snapshots),
        interval=1000 / max(config["attention_video_fps"], 1),
        blit=False,
    )

    video_path = config["attention_video_path"]

    if shutil.which("ffmpeg"):
        writer = animation.FFMpegWriter(fps=config["attention_video_fps"])
        anim.save(video_path, writer=writer, dpi=140)
    else:
        root, _ = os.path.splitext(video_path)
        video_path = root + ".gif"
        writer = animation.PillowWriter(fps=config["attention_video_fps"])
        anim.save(video_path, writer=writer, dpi=120)
        print("warning: ffmpeg not found; saved attention evolution as GIF instead of MP4.")

    plt.close(fig)
    print(f"saved attention evolution video: {video_path}")
    return video_path


# ============================
# Main
# ============================

models, optimizers = build_experiment_models()
teachers = build_ema_teachers(models) if config["use_self_distill"] else {}

init_csv_logger()
print_config_summary(models)

for model in models.values():
    model.train()

best_stats = {
    name: {
        "val_lm": float("inf"),
        "train_lm": float("inf"),
        "step": 0,
    }
    for name in models.keys()
}

attention_probe_x = None
attention_snapshots = []

if config["enable_attention_video"]:
    attention_probe_x, _ = get_batch("val")
    attention_probe_x = attention_probe_x[:config["attention_video_probe_batch"]]


for step in range(1, config["max_steps"] + 1):
    xb, yb = get_batch("train")

    step_logs = {}

    for name, model in models.items():
        if config["use_self_distill"]:
            step_logs[name] = train_one_model_step_self_distill(
                model=model,
                teacher=teachers[name],
                optimizer=optimizers[name],
            )
        else:
            step_logs[name] = train_one_model_step(
                model=model,
                optimizer=optimizers[name],
                xb=xb,
                yb=yb,
            )

    if step % config["eval_interval"] == 0 or step == 1:
        print_side_by_side_train(
            step=step,
            step_logs=step_logs,
            models=models,
        )

        stats = {
            name: estimate_loss(model)
            for name, model in models.items()
        }

        for name, s in stats.items():
            val_lm = s["val"]["lm"]

            if val_lm < best_stats[name]["val_lm"]:
                best_stats[name] = {
                    "val_lm": val_lm,
                    "train_lm": s["train"]["lm"],
                    "step": step,
                }

                save_best_checkpoint(
                    name=name,
                    model=models[name],
                    step=step,
                    best=best_stats[name],
                )

            append_csv_row(
                step=step,
                name=name,
                stats=s,
                best=best_stats[name],
                diag=models[name].get_attention_diagnostics(),
            )

        if config["enable_attention_video"] and attention_probe_x is not None:
            attention_snapshots.append(
                collect_attention_snapshot(
                    step=step,
                    models=models,
                    probe_x=attention_probe_x,
                )
            )

        print_side_by_side_eval(
            step=step,
            stats_by_model=stats,
            best_stats=best_stats,
            models=models,
        )

        print_samples(
            step=step,
            models=models,
        )


print("\n===== best checkpoints =====")
print("model       | best_train_lm | best_val_lm | best_step")
print("------------|---------------|-------------|----------")

for name in models.keys():
    best = best_stats[name]

    print(
        f"{name.upper():<11} | "
        f"{best['train_lm']:.4f}        | "
        f"{best['val_lm']:.4f}      | "
        f"{best['step']:>8d}"
    )


plot_training_artifacts()
build_attention_video(
    attention_snapshots=attention_snapshots,
    model_names=list(models.keys()),
)

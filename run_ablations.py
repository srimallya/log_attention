import argparse
import csv
import io
import os
import re
import shlex
import subprocess
import sys
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.py"
ABLATION_ROOT = ROOT / "ablation_checkpoints"


ABLATIONS = [
    {
        "name": "control_self_distill_alpha15",
        "overrides": {
            "use_self_distill": True,
            "use_alpha_clamp": True,
            "alpha_max": 1.5,
        },
    },
    {
        "name": "no_self_distill_alpha15",
        "overrides": {
            "use_self_distill": False,
            "use_alpha_clamp": True,
            "alpha_max": 1.5,
        },
    },
    {
        "name": "self_distill_alpha10",
        "overrides": {
            "use_self_distill": True,
            "use_alpha_clamp": True,
            "alpha_max": 1.0,
        },
    },
    {
        "name": "self_distill_alpha05",
        "overrides": {
            "use_self_distill": True,
            "use_alpha_clamp": True,
            "alpha_max": 0.5,
        },
    },
    {
        "name": "self_distill_no_alpha_clamp",
        "overrides": {
            "use_self_distill": True,
            "use_alpha_clamp": False,
        },
    },
    {
        "name": "no_self_distill_no_alpha_clamp",
        "overrides": {
            "use_self_distill": False,
            "use_alpha_clamp": False,
        },
    },
    # Optional seed ablations. Uncomment when needed.
    # {
    #     "name": "seed_2024_self_distill_alpha15",
    #     "overrides": {
    #         "use_self_distill": True,
    #         "use_alpha_clamp": True,
    #         "alpha_max": 1.5,
    #         "seed": 2024,
    #     },
    # },
    # {
    #     "name": "seed_42_self_distill_alpha15",
    #     "overrides": {
    #         "use_self_distill": True,
    #         "use_alpha_clamp": True,
    #         "alpha_max": 1.5,
    #         "seed": 42,
    #     },
    # },
]


SUMMARY_FIELDS = [
    "run_name",
    "best_dot_val_lm",
    "best_log_decay_val_lm",
    "best_gap_dot_minus_log_decay",
    "best_dot_step",
    "best_log_decay_step",
    "final_dot_val_lm",
    "final_log_decay_val_lm",
    "final_gap_dot_minus_log_decay",
    "mean_delta_after_5000",
    "fraction_log_decay_wins_after_5000",
    "final_log_decay_alpha_mean",
    "final_log_decay_alpha_max",
    "final_log_decay_mean_attn_dist",
    "final_log_decay_long_attn_mass",
]


def load_train_source():
    return TRAIN_PATH.read_text(encoding="utf-8")


def _line_offsets(source):
    offsets = []
    total = 0
    for line in source.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    return offsets


def _offset(line_offsets, position):
    line_no, col = position
    return line_offsets[line_no - 1] + col


def find_config_block(source):
    line_offsets = _line_offsets(source)
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    state = 0
    config_start = None
    depth = 0

    for token in tokens:
        tok_type = token.type
        tok_text = token.string

        if state == 0 and tok_type == tokenize.NAME and tok_text == "config":
            state = 1
            config_start = _offset(line_offsets, token.start)
        elif state == 1 and tok_text == "=":
            state = 2
        elif state == 2 and tok_type == tokenize.NAME and tok_text == "dict":
            state = 3
        elif state == 3 and tok_text == "(":
            depth = 1
            state = 4
        elif state == 4:
            if tok_text in "([{":
                depth += 1
            elif tok_text in ")]}":
                depth -= 1
                if depth == 0:
                    return config_start, _offset(line_offsets, token.end)
        elif tok_type not in {
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
        }:
            state = 0
            config_start = None

    raise ValueError(f"Could not find config = dict(...) block in {TRAIN_PATH}")


def format_python_value(value):
    if isinstance(value, Path):
        return repr(str(value))
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def _replace_config_line(line, key, value):
    pattern = re.compile(rf"^(\s*)({re.escape(key)})(\s*=\s*)(.*?)(,?)(\s*(?:#.*)?)$")
    match = pattern.match(line.rstrip("\n"))
    if not match:
        return None

    indent, name, sep, _old_value, comma, suffix = match.groups()
    comma = comma or ","
    newline = "\n" if line.endswith("\n") else ""
    return f"{indent}{name}{sep}{format_python_value(value)}{comma}{suffix}{newline}"


def apply_config_overrides(source, overrides):
    block_start, block_end = find_config_block(source)
    before = source[:block_start]
    block = source[block_start:block_end]
    after = source[block_end:]

    remaining = dict(overrides)
    new_lines = []

    for line in block.splitlines(keepends=True):
        replaced = None
        for key, value in list(remaining.items()):
            replaced = _replace_config_line(line, key, value)
            if replaced is not None:
                del remaining[key]
                break
        new_lines.append(replaced if replaced is not None else line)

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise KeyError(f"Config key(s) not found in train.py config block: {missing}")

    return before + "".join(new_lines) + after


def make_run_paths(run_name):
    run_dir = ABLATION_ROOT / run_name
    return {
        "run_dir": run_dir,
        "generated_train": run_dir / "train_generated.py",
        "log_path": run_dir / "run.log",
        "ckpt_dir": run_dir,
        "csv_log_path": run_dir / "metrics.csv",
        "plot_dir": run_dir / "plots",
        "attention_video_path": run_dir / "attention_evolution.mp4",
    }


def write_generated_train(run_name, overrides):
    paths = make_run_paths(run_name)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)

    isolated_overrides = {
        **overrides,
        "ckpt_dir": paths["ckpt_dir"],
        "csv_log_path": paths["csv_log_path"],
        "plot_dir": paths["plot_dir"],
        "attention_video_path": paths["attention_video_path"],
    }

    source = load_train_source()
    generated = apply_config_overrides(source, isolated_overrides)
    paths["generated_train"].write_text(generated, encoding="utf-8")
    return paths


def _stream_subprocess(command, log_path):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    quoted_command = " ".join(shlex.quote(part) for part in command)
    shell_command = (
        f"set -o pipefail; {quoted_command} 2>&1 | tee {shlex.quote(str(log_path))}"
    )
    result = subprocess.run(
        ["bash", "-lc", shell_command],
        cwd=str(ROOT),
        env=env,
        text=True,
        check=False,
    )
    return result.returncode


def run_ablation(run_name, overrides, args):
    effective_overrides = dict(overrides)
    if args.max_steps is not None:
        effective_overrides["max_steps"] = args.max_steps
    if args.disable_video:
        effective_overrides["enable_attention_video"] = False
    if args.disable_plots:
        effective_overrides["enable_post_training_plots"] = False

    paths = make_run_paths(run_name)
    if args.skip_existing and paths["csv_log_path"].exists():
        print(f"skip existing: {run_name} ({paths['csv_log_path']})")
        return 0

    paths = make_run_paths(run_name)
    command = [sys.executable, str(paths["generated_train"])]

    if args.dry_run:
        print(f"[dry-run] run: {run_name}")
        print(f"[dry-run] generated: {paths['generated_train']}")
        print(f"[dry-run] log: {paths['log_path']}")
        print(f"[dry-run] command: {' '.join(command)}")
        print(f"[dry-run] overrides: {effective_overrides}")
        return 0

    paths = write_generated_train(run_name, effective_overrides)
    print(f"\n===== running ablation: {run_name} =====")
    print(f"generated script: {paths['generated_train']}")
    print(f"log file: {paths['log_path']}")
    rc = _stream_subprocess(command, paths["log_path"])
    if rc != 0:
        print(f"ablation failed: {run_name} exited with {rc}")
    return rc


def _float_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def _rows_by_model(metrics_path):
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_model = {}
    for row in rows:
        model = row.get("model", "")
        if not model:
            continue
        row["_step"] = _int_or_none(row.get("step"))
        row["_val_lm"] = _float_or_none(row.get("val_lm"))
        by_model.setdefault(model, []).append(row)

    for model_rows in by_model.values():
        model_rows.sort(key=lambda row: row["_step"] if row["_step"] is not None else -1)

    return by_model


def _best_row(rows):
    valid = [row for row in rows if row.get("_val_lm") is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: row["_val_lm"])


def _final_row(rows):
    valid = [row for row in rows if row.get("_step") is not None]
    if not valid:
        return rows[-1] if rows else None
    return max(valid, key=lambda row: row["_step"])


def _paired_delta_stats(dot_rows, log_rows, min_step=5000):
    dot_by_step = {
        row["_step"]: row["_val_lm"]
        for row in dot_rows
        if row.get("_step") is not None and row.get("_val_lm") is not None
    }
    log_by_step = {
        row["_step"]: row["_val_lm"]
        for row in log_rows
        if row.get("_step") is not None and row.get("_val_lm") is not None
    }
    common_steps = sorted(step for step in dot_by_step.keys() & log_by_step.keys() if step >= min_step)
    if not common_steps:
        return None, None

    deltas = [dot_by_step[step] - log_by_step[step] for step in common_steps]
    wins = [delta > 0 for delta in deltas]
    return sum(deltas) / len(deltas), sum(wins) / len(wins)


def summarize_one_run(run_name, metrics_path):
    by_model = _rows_by_model(metrics_path)
    dot_rows = by_model.get("dot", [])
    log_rows = by_model.get("log_decay", [])

    best_dot = _best_row(dot_rows)
    best_log = _best_row(log_rows)
    final_dot = _final_row(dot_rows)
    final_log = _final_row(log_rows)

    best_dot_val = best_dot["_val_lm"] if best_dot else None
    best_log_val = best_log["_val_lm"] if best_log else None
    final_dot_val = final_dot["_val_lm"] if final_dot else None
    final_log_val = final_log["_val_lm"] if final_log else None
    mean_delta, fraction_wins = _paired_delta_stats(dot_rows, log_rows)

    return {
        "run_name": run_name,
        "best_dot_val_lm": best_dot_val,
        "best_log_decay_val_lm": best_log_val,
        "best_gap_dot_minus_log_decay": (
            best_dot_val - best_log_val
            if best_dot_val is not None and best_log_val is not None
            else None
        ),
        "best_dot_step": best_dot["_step"] if best_dot else None,
        "best_log_decay_step": best_log["_step"] if best_log else None,
        "final_dot_val_lm": final_dot_val,
        "final_log_decay_val_lm": final_log_val,
        "final_gap_dot_minus_log_decay": (
            final_dot_val - final_log_val
            if final_dot_val is not None and final_log_val is not None
            else None
        ),
        "mean_delta_after_5000": mean_delta,
        "fraction_log_decay_wins_after_5000": fraction_wins,
        "final_log_decay_alpha_mean": _float_or_none(final_log.get("alpha_mean")) if final_log else None,
        "final_log_decay_alpha_max": _float_or_none(final_log.get("alpha_max")) if final_log else None,
        "final_log_decay_mean_attn_dist": (
            _float_or_none(final_log.get("mean_attn_dist")) if final_log else None
        ),
        "final_log_decay_long_attn_mass": (
            _float_or_none(final_log.get("long_attn_mass")) if final_log else None
        ),
    }


def write_summary_plots(summary_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping summary plots.")
        return []

    plot_dir = ABLATION_ROOT / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def bar_plot(filename, field, title, ylabel):
        rows = [row for row in summary_rows if row.get(field) not in (None, "")]
        if not rows:
            return
        names = [row["run_name"] for row in rows]
        values = [float(row[field]) for row in rows]
        fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.1), 5))
        ax.bar(range(len(names)), values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        out_path = plot_dir / filename
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        written.append(out_path)

    bar_plot("best_val_by_run.png", "best_log_decay_val_lm", "Best log-decay validation LM", "val_lm")
    bar_plot("gap_by_run.png", "best_gap_dot_minus_log_decay", "Best gap: dot - log_decay", "val_lm gap")
    bar_plot(
        "mean_delta_after_5000_by_run.png",
        "mean_delta_after_5000",
        "Mean paired delta after step 5000",
        "dot val_lm - log_decay val_lm",
    )
    bar_plot("alpha_mean_by_run.png", "final_log_decay_alpha_mean", "Final log-decay alpha mean", "alpha")

    return written


def summarize_results():
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for metrics_path in sorted(ABLATION_ROOT.glob("*/metrics.csv")):
        run_name = metrics_path.parent.name
        try:
            rows.append(summarize_one_run(run_name, metrics_path))
        except Exception as exc:
            print(f"warning: could not summarize {metrics_path}: {exc}")

    summary_path = ABLATION_ROOT / "ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in SUMMARY_FIELDS})

    written_plots = write_summary_plots(rows)
    print(f"summary written: {summary_path}")
    if written_plots:
        print(f"summary plots written: {ABLATION_ROOT / 'summary_plots'}")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run isolated ablations by generating per-run train.py copies."
    )
    parser.add_argument("--only", help="Run only one named ablation.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its metrics.csv already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated config paths and commands without executing.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max_steps for all ablations.",
    )
    parser.add_argument(
        "--disable-video",
        action="store_true",
        help="Override enable_attention_video=False for all ablations.",
    )
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Override enable_post_training_plots=False for all ablations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected = ABLATIONS

    if args.only:
        selected = [ablation for ablation in ABLATIONS if ablation["name"] == args.only]
        if not selected:
            names = ", ".join(ablation["name"] for ablation in ABLATIONS)
            raise SystemExit(f"Unknown ablation {args.only!r}. Available: {names}")

    failures = []
    for ablation in selected:
        rc = run_ablation(ablation["name"], ablation["overrides"], args)
        if rc != 0:
            failures.append((ablation["name"], rc))
            break

    if not args.dry_run:
        summarize_results()

    if failures:
        failed = ", ".join(f"{name} ({rc})" for name, rc in failures)
        raise SystemExit(f"ablation failures: {failed}")


if __name__ == "__main__":
    main()

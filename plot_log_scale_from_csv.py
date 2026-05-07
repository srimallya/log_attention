import argparse
import csv
import os


SUMMARY_FIELDS = [
    "model",
    "late_step_min",
    "eval_count",
    "mean_val_lm",
    "best_val_lm",
    "best_val_step",
    "mean_attn_dist",
    "mean_long_attn_mass",
    "final_val_lm",
    "final_step",
    "final_mean_attn_dist",
    "final_long_attn_mass",
    "final_alpha_mean",
]


def parse_float(value):
    if value is None or value == "":
        return None

    return float(value)


def load_metric_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
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


def save_plot(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path


def set_positive_log_y(ax):
    bottom, top = ax.get_ylim()

    if top > 0:
        ax.set_yscale("log")

        if bottom <= 0:
            ax.set_ylim(bottom=max(top * 1e-6, 1e-12))


def set_positive_log_x(ax):
    left, right = ax.get_xlim()

    if right > 0:
        ax.set_xscale("log")

        if left <= 0:
            ax.set_xlim(left=max(right * 1e-6, 1e-12))


def infer_layer_count(rows):
    layer_ids = set()

    for row in rows:
        for key in row:
            if key.startswith("L") and key.endswith("_alpha_mean"):
                layer_id = key[1:].split("_", 1)[0]

                if layer_id.isdigit():
                    layer_ids.add(int(layer_id))

    if not layer_ids:
        return 0

    return max(layer_ids) + 1


def mean_or_none(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def fmt(value):
    if value is None:
        return ""

    if isinstance(value, int):
        return str(value)

    return f"{value:.6f}"


def effective_summary_step_min(rows, late_step_min):
    if any(row["step"] >= late_step_min for row in rows):
        return late_step_min

    return min(row["step"] for row in rows)


def late_stage_summary(rows, late_step_min):
    effective_step_min = effective_summary_step_min(rows, late_step_min)
    model_names = sorted(set(row["model"] for row in rows))
    by_model = {name: rows_for_model(rows, name) for name in model_names}
    summaries = []

    for name, model_rows in by_model.items():
        late_rows = [row for row in model_rows if row["step"] >= effective_step_min]

        if not late_rows:
            continue

        best_row = min(late_rows, key=lambda row: row["val_lm"])
        final_row = max(late_rows, key=lambda row: row["step"])

        summaries.append({
            "model": name,
            "late_step_min": effective_step_min,
            "eval_count": len(late_rows),
            "mean_val_lm": mean_or_none([row.get("val_lm") for row in late_rows]),
            "best_val_lm": best_row.get("val_lm"),
            "best_val_step": best_row.get("step"),
            "mean_attn_dist": mean_or_none([
                row.get("mean_attn_dist")
                for row in late_rows
            ]),
            "mean_long_attn_mass": mean_or_none([
                row.get("long_attn_mass")
                for row in late_rows
            ]),
            "final_val_lm": final_row.get("val_lm"),
            "final_step": final_row.get("step"),
            "final_mean_attn_dist": final_row.get("mean_attn_dist"),
            "final_long_attn_mass": final_row.get("long_attn_mass"),
            "final_alpha_mean": final_row.get("alpha_mean"),
        })

    comparisons = {}

    if "dot" in by_model and "log_decay" in by_model:
        dot_by_step = {
            row["step"]: row
            for row in by_model["dot"]
            if row["step"] >= effective_step_min
        }
        log_by_step = {
            row["step"]: row
            for row in by_model["log_decay"]
            if row["step"] >= effective_step_min
        }
        common_steps = sorted(set(dot_by_step) & set(log_by_step))
        deltas = [
            log_by_step[step]["val_lm"] - dot_by_step[step]["val_lm"]
            for step in common_steps
        ]

        if deltas:
            comparisons = {
                "late_step_min": effective_step_min,
                "common_eval_count": len(common_steps),
                "log_decay_win_fraction": sum(
                    1 for delta in deltas if delta < 0
                ) / len(deltas),
                "mean_delta_log_decay_minus_dot": sum(deltas) / len(deltas),
                "best_delta_log_decay_minus_dot": min(deltas),
                "worst_delta_log_decay_minus_dot": max(deltas),
            }

    return summaries, comparisons, effective_step_min


def save_late_stage_summary(rows, output_dir, late_step_min):
    os.makedirs(output_dir, exist_ok=True)

    summaries, comparisons, effective_step_min = late_stage_summary(
        rows,
        late_step_min,
    )
    csv_path = os.path.join(output_dir, "late_stage_summary.csv")
    md_path = os.path.join(output_dir, "late_stage_summary.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for row in summaries:
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Late-Stage Summary\n\n")
        f.write(f"`requested_late_step_min = {late_step_min}`\n\n")
        f.write(f"`effective_step_min = {effective_step_min}`\n\n")

        if effective_step_min != late_step_min:
            f.write(
                "The requested late-stage threshold is beyond the available "
                "CSV steps, so this summary uses all available eval rows.\n\n"
            )

        if summaries:
            f.write("| model | evals | mean val_lm | best val_lm | best step | mean attention distance | mean long attention mass | final val_lm | final mean attention distance | final long attention mass | final alpha mean |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")

            for row in summaries:
                f.write(
                    f"| {row['model']} "
                    f"| {row['eval_count']} "
                    f"| {fmt(row['mean_val_lm'])} "
                    f"| {fmt(row['best_val_lm'])} "
                    f"| {fmt(row['best_val_step'])} "
                    f"| {fmt(row['mean_attn_dist'])} "
                    f"| {fmt(row['mean_long_attn_mass'])} "
                    f"| {fmt(row['final_val_lm'])} "
                    f"| {fmt(row['final_mean_attn_dist'])} "
                    f"| {fmt(row['final_long_attn_mass'])} "
                    f"| {fmt(row['final_alpha_mean'])} |\n"
                )

        if comparisons:
            f.write("\n## DOT vs LOG_DECAY\n\n")
            f.write(
                f"- Common late evals: {comparisons['common_eval_count']}\n"
            )
            f.write(
                "- Fraction where `log_decay` beats `dot`: "
                f"{comparisons['log_decay_win_fraction']:.4f}\n"
            )
            f.write(
                "- Mean delta, `log_decay - dot`: "
                f"{comparisons['mean_delta_log_decay_minus_dot']:.6f}\n"
            )
            f.write(
                "- Best delta, `log_decay - dot`: "
                f"{comparisons['best_delta_log_decay_minus_dot']:.6f}\n"
            )
            f.write(
                "- Worst delta, `log_decay - dot`: "
                f"{comparisons['worst_delta_log_decay_minus_dot']:.6f}\n"
            )

    return [csv_path, md_path]


def plot_log_scale_training_artifacts(csv_path, output_dir, late_step_min=5000):
    import matplotlib.pyplot as plt

    rows = load_metric_rows(csv_path)

    if not rows:
        raise ValueError(f"No metric rows found in {csv_path}")

    model_names = sorted(set(row["model"] for row in rows))
    by_model = {name: rows_for_model(rows, name) for name in model_names}
    n_layer = infer_layer_count(rows)
    effective_step_min = effective_summary_step_min(rows, late_step_min)
    written = save_late_stage_summary(rows, output_dir, late_step_min)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "train_lm")
        ax.plot(xs, ys, linestyle="--", label=f"{name} train")
        xs, ys = series(model_rows, "val_lm")
        ax.plot(xs, ys, label=f"{name} val")
    ax.set_title("Train and validation LM loss - log scale")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    set_positive_log_y(ax)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    written.append(save_plot(fig, output_dir, "01_train_val_lm_loss_log.png"))
    plt.close(fig)

    if "dot" in by_model and "log_decay" in by_model:
        dot_by_step = {row["step"]: row["val_lm"] for row in by_model["dot"]}
        log_by_step = {row["step"]: row["val_lm"] for row in by_model["log_decay"]}
        common_steps = sorted(set(dot_by_step) & set(log_by_step))
        deltas = [log_by_step[step] - dot_by_step[step] for step in common_steps]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.plot(common_steps, deltas, label="log_decay - dot")
        ax.set_title("Validation loss delta - symmetric log scale")
        ax.set_xlabel("step")
        ax.set_ylabel("delta loss")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend()
        written.append(save_plot(fig, output_dir, "02_val_loss_delta_symlog.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "best_val_lm")
        ax.plot(xs, ys, label=name)
    ax.set_title("Best validation LM loss - log scale")
    ax.set_xlabel("step")
    ax.set_ylabel("best val loss")
    set_positive_log_y(ax)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    written.append(save_plot(fig, output_dir, "03_best_val_lm_log.png"))
    plt.close(fig)

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

            ax.set_title("Log-decay alpha evolution - log scale")
            ax.set_xlabel("step")
            ax.set_ylabel("alpha")
            set_positive_log_y(ax)
            ax.grid(True, alpha=0.25, which="both")
            ax.legend()
            written.append(save_plot(fig, output_dir, "04_alpha_evolution_log.png"))
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for layer_idx in range(n_layer):
            xs, ys = series(log_rows, f"L{layer_idx}_alpha_mean")
            if xs and ys:
                plotted = True
                ax.plot(xs, ys, label=f"L{layer_idx}")

        if plotted:
            ax.set_title("Layer-wise log-decay alpha - log scale")
            ax.set_xlabel("step")
            ax.set_ylabel("alpha")
            set_positive_log_y(ax)
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(ncol=2)
            written.append(save_plot(fig, output_dir, "05_layerwise_alpha_log.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "long_attn_mass")
        if xs and ys:
            ax.plot(xs, ys, label=name)
    ax.set_title("Long-range attention mass - log scale")
    ax.set_xlabel("step")
    ax.set_ylabel("attention mass")
    set_positive_log_y(ax)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    written.append(save_plot(fig, output_dir, "06_long_range_attention_mass_log.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "mean_attn_dist")
        if xs and ys:
            ax.plot(xs, ys, label=name)
    ax.set_title("Mean attention distance - log scale")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens")
    set_positive_log_y(ax)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    written.append(save_plot(fig, output_dir, "07_mean_attention_distance_log.png"))
    plt.close(fig)

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
    ax.set_title("Validation loss vs mean attention distance - log scale")
    ax.set_xlabel("mean attention distance")
    ax.set_ylabel("val LM loss")
    set_positive_log_x(ax)
    set_positive_log_y(ax)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    written.append(save_plot(fig, output_dir, "08_val_loss_vs_attention_distance_loglog.png"))
    plt.close(fig)

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
        ax.set_title("Best validation LM loss by model - log scale")
        ax.set_xlabel("model")
        ax.set_ylabel("best val LM loss")
        set_positive_log_y(ax)
        ax.grid(True, axis="y", alpha=0.25, which="both")
        written.append(save_plot(fig, output_dir, "09_best_val_lm_bar_log.png"))
        plt.close(fig)

        margin = 0.01
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, values)
        ax.set_title("Best validation LM loss by model - zoomed")
        ax.set_xlabel("model")
        ax.set_ylabel("best val LM loss")
        ax.set_ylim(min(values) - margin, max(values) + margin)
        ax.grid(True, axis="y", alpha=0.25)
        written.append(save_plot(fig, output_dir, "10_best_val_lm_bar_zoomed.png"))
        plt.close(fig)

    if "dot" in by_model:
        dot_best = min(
            row["best_val_lm"]
            for row in by_model["dot"]
            if row.get("best_val_lm") is not None
        )
        improvement_labels = []
        improvement_values = []

        for name, model_rows in by_model.items():
            best_values = [
                row["best_val_lm"]
                for row in model_rows
                if row.get("best_val_lm") is not None
            ]

            if best_values:
                improvement_labels.append(name)
                improvement_values.append(dot_best - min(best_values))

        if improvement_labels and improvement_values:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.axhline(0.0, color="black", linewidth=1)
            ax.bar(improvement_labels, improvement_values)
            ax.set_title("Best validation improvement over DOT")
            ax.set_xlabel("model")
            ax.set_ylabel("DOT best val_lm - model best val_lm")
            ax.grid(True, axis="y", alpha=0.25)
            written.append(save_plot(fig, output_dir, "11_best_val_improvement_over_dot.png"))
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    marker_by_model = {
        "dot": "o",
        "log_decay": "x",
    }
    scatter = None

    for name, model_rows in by_model.items():
        xs = []
        ys = []
        colors = []

        for row in model_rows:
            if row.get("mean_attn_dist") is not None and row.get("val_lm") is not None:
                xs.append(row["mean_attn_dist"])
                ys.append(row["val_lm"])
                colors.append(row["step"])

        if xs and ys:
            scatter = ax.scatter(
                xs,
                ys,
                c=colors,
                s=34,
                alpha=0.78,
                marker=marker_by_model.get(name, "s"),
                label=name,
            )

    ax.set_title("Validation loss vs mean attention distance - colored by step")
    ax.set_xlabel("mean attention distance")
    ax.set_ylabel("val LM loss")
    set_positive_log_x(ax)
    set_positive_log_y(ax)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="step")
    written.append(save_plot(fig, output_dir, "12_val_loss_vs_attention_distance_by_step.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    for name, model_rows in by_model.items():
        xs = []
        ys = []

        for row in model_rows:
            if (
                row["step"] >= effective_step_min
                and row.get("mean_attn_dist") is not None
                and row.get("val_lm") is not None
            ):
                xs.append(row["mean_attn_dist"])
                ys.append(row["val_lm"])

        if xs and ys:
            plotted = True
            ax.scatter(xs, ys, s=26, alpha=0.75, label=name)

    if plotted:
        ax.set_title(
            f"Validation loss vs mean attention distance - step >= {effective_step_min}"
        )
        ax.set_xlabel("mean attention distance")
        ax.set_ylabel("val LM loss")
        set_positive_log_x(ax)
        set_positive_log_y(ax)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend()
        written.append(save_plot(fig, output_dir, "13_val_loss_vs_attention_distance_late.png"))
    plt.close(fig)

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Generate log-scale plot artifacts from an existing metrics.csv file.",
    )
    parser.add_argument(
        "--csv",
        default="checkpoints_log_decay/metrics.csv",
        help="Path to metrics.csv produced by train.py.",
    )
    parser.add_argument(
        "--out",
        default="checkpoints_log_decay/plots_log_scale",
        help="Directory for generated log-scale plot PNG files.",
    )
    parser.add_argument(
        "--late-step-min",
        default=5000,
        type=int,
        help="Minimum step for the late-stage scatter plot.",
    )
    args = parser.parse_args()

    written = plot_log_scale_training_artifacts(
        args.csv,
        args.out,
        late_step_min=args.late_step_min,
    )

    print(f"saved {len(written)} log-scale plot artifacts to {args.out}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

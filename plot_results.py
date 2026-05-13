import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(csv_path):
    rows = []
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append({
                "model": row["model"],
                "train_set_size": int(row.get("train_set_size", row.get("set_size"))),
                "test_set_size": int(row.get("test_set_size", row.get("set_size"))),
                "process_steps": row["process_steps"],
                "glimpses": int(row.get("glimpses", 1)),
                "exact_match": float(row["exact_match"]),
                "valid_permutation": float(row["valid_permutation"]),
                "train_loss": float(row["train_loss"]),
                "train_seconds": float(row.get("train_seconds", row.get("seconds", 0.0))),
            })
    return rows


def group_by_test_size(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["test_set_size"], []).append(row)
    return dict(sorted(grouped.items()))


def get_baseline(rows, glimpses):
    for row in rows:
        if row["model"] == "PointerNetwork" and row["glimpses"] == glimpses:
            return row
    return None


def get_rpw_rows(rows, glimpses):
    rpw_rows = [
        row for row in rows
        if row["model"] == "ReadProcessWrite" and row["glimpses"] == glimpses
    ]
    return sorted(rpw_rows, key=lambda row: int(row["process_steps"]))


def plot_metric_by_process_steps(grouped, metric, ylabel, output_path):
    fig, axes = plt.subplots(
        1,
        len(grouped),
        figsize=(5 * len(grouped), 4),
        sharey=True
    )
    if len(grouped) == 1:
        axes = [axes]

    glimpse_styles = {
        0: {"color": "tab:blue", "marker": "o", "label": "RPW, glimpses=0"},
        1: {"color": "tab:green", "marker": "s", "label": "RPW, glimpses=1"},
    }
    baseline_styles = {
        0: {"color": "tab:orange", "label": "Ptr-Net, glimpses=0"},
        1: {"color": "tab:red", "label": "Ptr-Net, glimpses=1"},
    }

    for ax, (test_set_size, rows) in zip(axes, grouped.items()):
        for glimpses, style in glimpse_styles.items():
            rpw_rows = get_rpw_rows(rows, glimpses)
            if not rpw_rows:
                continue

            process_steps = [int(row["process_steps"]) for row in rpw_rows]
            values = [row[metric] for row in rpw_rows]
            ax.plot(
                process_steps,
                values,
                marker=style["marker"],
                linewidth=2,
                color=style["color"],
                label=style["label"]
            )

            for x, y in zip(process_steps, values):
                ax.annotate(
                    f"{y:.3f}",
                    xy=(x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )

        for glimpses, style in baseline_styles.items():
            baseline = get_baseline(rows, glimpses)
            if baseline is None:
                continue

            ax.axhline(
                baseline[metric],
                linestyle="--",
                linewidth=2,
                color=style["color"],
                label=style["label"]
            )

        all_steps = sorted({
            int(row["process_steps"])
            for row in rows
            if row["model"] == "ReadProcessWrite"
        })
        ax.set_title(f"Train N = {rows[0]['train_set_size']}, Test N = {test_set_size}")
        ax.set_xlabel("Process steps P")
        ax.set_xticks(all_steps)
        ax.grid(True, alpha=0.3)

        if metric in {"exact_match", "valid_permutation"}:
            ax.set_ylim(0, 1.05)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_paper_style_accuracy(grouped, output_path):
    labels = []
    values = []
    colors = []

    for test_set_size, rows in grouped.items():
        for glimpses in (0, 1):
            baseline = get_baseline(rows, glimpses)
            if baseline is not None:
                labels.append(f"Ptr\nG={glimpses}\nN={test_set_size}")
                values.append(baseline["exact_match"])
                colors.append("tab:orange" if glimpses == 0 else "tab:red")

        for process_steps in sorted({
            int(row["process_steps"])
            for row in rows
            if row["model"] == "ReadProcessWrite"
        }):
            for glimpses in (0, 1):
                matches = [
                    row for row in rows
                    if row["model"] == "ReadProcessWrite"
                    and int(row["process_steps"]) == process_steps
                    and row["glimpses"] == glimpses
                ]
                if not matches:
                    continue
                row = matches[0]
                labels.append(f"P={process_steps}\nG={glimpses}\nN={test_set_size}")
                values.append(row["exact_match"])
                colors.append("tab:blue" if glimpses == 0 else "tab:green")

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Exact match accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sorting accuracy by test length, process steps, and glimpses")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            fontsize=7,
            rotation=90 if len(labels) > 24 else 0
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot experiment_results.csv as summary figures."
    )
    parser.add_argument("--input", default="experiment_results.csv")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf"],
        default="png"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(args.input)
    grouped = group_by_test_size(rows)
    suffix = args.format

    plot_metric_by_process_steps(
        grouped,
        metric="exact_match",
        ylabel="Exact match accuracy",
        output_path=output_dir / f"accuracy_by_process_steps.{suffix}"
    )
    plot_metric_by_process_steps(
        grouped,
        metric="train_seconds",
        ylabel="Training time (seconds)",
        output_path=output_dir / f"time_by_process_steps.{suffix}"
    )
    plot_metric_by_process_steps(
        grouped,
        metric="train_loss",
        ylabel="Final train loss",
        output_path=output_dir / f"loss_by_process_steps.{suffix}"
    )
    plot_paper_style_accuracy(
        grouped,
        output_path=output_dir / f"paper_style_accuracy.{suffix}"
    )

    print(f"Saved {suffix} figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

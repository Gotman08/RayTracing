#!/usr/bin/env python3
"""Regenerate SVG figures and summary statistics from the committed raw CSV."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", default="bench/results/wsl-2026-09-12")
    parser.add_argument("--preview-dir", help="Optional PNG previews for visual inspection")
    args = parser.parse_args()
    directory = ROOT / args.results
    with (directory / "raw.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("raw.csv contains no observations")
    threads = sorted({int(row["threads"]) for row in rows})
    if len(threads) != 1:
        raise ValueError("A figure must use one CPU thread count")
    groups = {}
    for row in rows:
        if int(row["exit_code"]) != 0:
            raise ValueError("Raw data contains a failed render")
        if int(row["warmup"]):
            continue
        key = (row["mode"], int(row["width"]), int(row["height"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (mode, width, height), observations in sorted(groups.items()):
        if len(observations) != 5:
            raise ValueError(f"Expected five measured trials for {mode} {width}x{height}")
        entry = dict(mode=mode, width=width, height=height, trials=len(observations))
        for metric in ["render_ms", "wall_ms"]:
            values = [float(row[metric]) for row in observations]
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
            entry[metric] = dict(median=statistics.median(values), q1=q1, q3=q3, iqr=q3-q1)
        summary.append(entry)
    (directory / "summary.json").write_text(json.dumps({
        "quartile_method": "statistics.quantiles(method='inclusive')",
        "warmups_excluded": True,
        "results": summary,
    }, indent=2) + "\n")
    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "raytracing-benchmark-v1", "svg.fonttype": "none"})
    sizes = sorted({(row["width"], row["height"]) for row in summary})
    modes = sorted({row["mode"] for row in summary})
    for theme, background, foreground, grid, colors in [
        ("light", "#ffffff", "#17212b", "#d8dee5", ["#286d8a", "#ae5930"]),
        ("dark", "#101820", "#e5edf4", "#3a4650", ["#75c2e2", "#f1ac82"]),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), layout="constrained")
        fig.set_facecolor(background)
        for ax, metric, title in zip(axes, ["wall_ms", "render_ms"],
                                     ["Full process, including export", "Reported render stage"]):
            ax.set_facecolor(background)
            for mode_index, mode in enumerate(modes):
                entries = [next(row for row in summary if (row["mode"], row["width"], row["height"]) == (mode, *size)) for size in sizes]
                x = [index + (mode_index - (len(modes)-1)/2)*0.15 for index in range(len(sizes))]
                medians = [row[metric]["median"] for row in entries]
                low = [row[metric]["median"]-row[metric]["q1"] for row in entries]
                high = [row[metric]["q3"]-row[metric]["median"] for row in entries]
                ax.errorbar(x, medians, yerr=[low, high], marker="o", linestyle="none",
                            markersize=6, capsize=5, color=colors[mode_index],
                            label=f"CPU, {threads[0]} threads" if mode == "cpu" else "CUDA GPU")
                for position, median in zip(x, medians):
                    ax.annotate(f"{median:,.1f}", (position, median), xytext=(0, 12),
                                textcoords="offset points", ha="center", color=colors[mode_index], fontsize=9)
            ax.set_yscale("log")
            values = [row[metric]["median"] for row in summary]
            ax.set_ylim(min(values)*0.45, max(values)*2.3)
            ax.set_xlim(-0.45, len(sizes)-0.55)
            ax.set_xticks(range(len(sizes)), [f"{w} × {h}" for w, h in sizes])
            ax.set_xlabel("Image size (pixels)", color=foreground)
            ax.set_ylabel("Time (ms, logarithmic scale)", color=foreground)
            ax.set_title(title, color=foreground, pad=14, fontsize=11)
            ax.tick_params(colors=foreground)
            ax.grid(axis="y", which="major", color=grid, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            for side in ["left", "bottom"]:
                ax.spines[side].set_color(grid)
            ax.legend(frameon=False, labelcolor=foreground, loc="lower right")
        fig.supxlabel("50 samples/pixel, depth 20. Median and Q1–Q3 of five trials; one warmup excluded.\nRender stages differ: CPU includes tone mapping; GPU shows the HDR sampling kernel.",
                      color=foreground, fontsize=9)
        fig.savefig(assets / f"bench-{theme}.svg", metadata={"Date": None}, facecolor=background)
        if args.preview_dir:
            preview = Path(args.preview_dir)
            preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview / f"raytracing-bench-{theme}.png", dpi=160, facecolor=background)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

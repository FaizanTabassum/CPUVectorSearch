#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROUTE_COLOURS = {
    "scalar_flat": "#2a78d6",
    "optimized_flat": "#eb6834",
    "ivf": "#1baf7a",
}
EXTRA_COLOURS = ["#eda100", "#e87ba4", "#e34948"]
IVF_LIST_COLOURS = ["#1baf7a", "#6250d6", "#008300"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
TARGET_RECALL_DEFAULT = 0.95
DESCRIPTION = "Turn a run_benchmarks.py output folder into summary tables and figures."


def number(value):
    if value in ("", None, "None", "null"):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def median(values):
    values = [v for v in values if isinstance(v, float)]
    return statistics.median(values) if values else None


def load_rows(run_dir):
    path = Path(run_dir) / "summary.csv"
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run scripts/run_benchmarks.py first")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("error: summary.csv has no runs")
    return rows


def aggregate(rows):
    by_config = defaultdict(list)
    for row in rows:
        by_config[(row["route"], row["params"], int(row["database_rows"]))].append(row)

    per_config = []
    for (route, params, size), group in by_config.items():
        keys = [k for k in group[0] if k not in ("run", "process_run", "params", "route",
                                                  "variant", "backend", "split", "threads_run")]
        merged = {"route": route, "params": params, "database_rows": size,
                  "backend": group[0]["backend"], "seed": number(group[0]["seed"]),
                  "lists": number(group[0]["lists"]), "probes": number(group[0]["probes"]),
                  "process_runs": len(group),
                  "threaded_results_match": all(r["threaded_results_match"] == "True" for r in group)}
        for key in keys:
            if key in merged:
                continue
            merged[key] = median([number(r[key]) for r in group])
        per_config.append(merged)

    by_setting = defaultdict(list)
    for row in per_config:
        setting = ";".join(p for p in row["params"].split(";") if p and not p.startswith("seed="))
        by_setting[(row["route"], setting, row["database_rows"])].append(row)
    final = []
    for (route, setting, size), group in sorted(by_setting.items(),
                                                key=lambda item: (item[0][2], item[0][0], item[0][1])):
        row = dict(group[0])
        row["params"] = setting
        row["seeds"] = len(group)
        for key in group[0]:
            values = [g[key] for g in group]
            if all(isinstance(v, float) for v in values) and len(values) > 1:
                row[key] = statistics.fmean(values)
        if len(group) > 1:
            recalls = [g["recall_at_k"] for g in group if isinstance(g["recall_at_k"], float)]
            row["recall_seed_min"] = min(recalls) if recalls else None
            row["recall_seed_max"] = max(recalls) if recalls else None
        row["variant"] = route + (f" ({setting})" if setting else "")
        row["threaded_results_match"] = all(g["threaded_results_match"] for g in group)
        final.append(row)
    return final


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}"
    return str(value)


def mib(value):
    return None if value is None else value / (1024 * 1024)


def best_ivf(rows, size, target):
    passing = [r for r in rows if r["route"] == "ivf" and r["database_rows"] == size
               and (r.get("recall_at_k") or 0) >= target - 1e-9]
    return min(passing, key=lambda r: r["latency_median_ms"]) if passing else None


def headline_rows(rows, target):
    out = [r for r in rows if r["route"] != "ivf"]
    for size in sorted({r["database_rows"] for r in rows}):
        chosen = best_ivf(rows, size, target)
        if chosen:
            out.append(chosen)
    return sorted(out, key=lambda r: (r["database_rows"], r["route"], r["params"]))


def thread_counts(rows):
    return sorted({int(k[len("qps_t"):]) for r in rows for k in r if k.startswith("qps_t")})


def write_tables(rows, out_dir, target):
    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k not in ("variant", "database_rows"), k))
    with (out_dir / "aggregated.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    threads = thread_counts(rows)
    lines = ["# Benchmark summary", "",
             "Medians over process runs; IVF figures are averaged over training seeds.",
             f"IVF rows show the fastest setting with mean recall@10 >= {target}.", "",
             "## Single-thread latency, recall and build", "",
             "| Rows | Route | Recall@10 | Median ms | p95 ms | p99 ms | Candidates | Build ms |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in headline_rows(rows, target):
        lines.append(f"| {r['database_rows']:,} | {r['variant']} | {fmt(r.get('recall_at_k'), 4)} | "
                     f"{fmt(r.get('latency_median_ms'))} | {fmt(r.get('latency_p95_ms'))} | "
                     f"{fmt(r.get('latency_p99_ms'))} | {fmt(r.get('candidates_per_query'), 0)} | "
                     f"{fmt(r.get('index_build_ms'))} |")
    lines += ["", "## Batch throughput (queries per second)", "",
              "| Rows | Route | " + " | ".join(f"{t} thread{'s' if t > 1 else ''}" for t in threads)
              + " | Best speedup | Same results |",
              "| ---: | --- | " + " | ".join("---:" for _ in threads) + " | ---: | --- |"]
    for r in headline_rows(rows, target):
        speedups = [r.get(f"speedup_t{t}") for t in threads if r.get(f"speedup_t{t}")]
        lines.append(f"| {r['database_rows']:,} | {r['variant']} | "
                     + " | ".join(fmt(r.get(f"qps_t{t}"), 0) for t in threads)
                     + f" | {fmt(max(speedups) if speedups else None, 2)}x | "
                     f"{fmt(r['threaded_results_match'])} |")
    lines += ["", "## Memory (MiB)", "",
              "| Rows | Route | Peak RSS | RSS after build | Index RSS delta |",
              "| ---: | --- | ---: | ---: | ---: |"]
    for r in headline_rows(rows, target):
        lines.append(f"| {r['database_rows']:,} | {r['variant']} | {fmt(mib(r.get('peak_memory_bytes')), 1)} | "
                     f"{fmt(mib(r.get('rss_after_build_bytes')), 1)} | "
                     f"{fmt(mib(r.get('index_rss_delta_bytes')), 2)} |")
    ivf = [r for r in rows if r["route"] == "ivf"]
    if ivf:
        lines += ["", "## IVF sweep (all settings)", "",
                  "| Rows | Lists | Probes | Recall@10 (mean of seeds) | Seed min-max | Median ms | Candidates |",
                  "| ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
        for r in sorted(ivf, key=lambda r: (r["database_rows"], r["lists"] or 0, r["probes"] or 0)):
            spread = (f"{fmt(r.get('recall_seed_min'), 4)}-{fmt(r.get('recall_seed_max'), 4)}"
                      if r.get("recall_seed_min") is not None else "n/a")
            lines.append(f"| {r['database_rows']:,} | {fmt(r['lists'], 0)} | {fmt(r['probes'], 0)} | "
                         f"{fmt(r.get('recall_at_k'), 4)} | {spread} | {fmt(r.get('latency_median_ms'))} | "
                         f"{fmt(r.get('candidates_per_query'), 0)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def colour_for(key, assigned):
    if key in ROUTE_COLOURS:
        return ROUTE_COLOURS[key]
    if key not in assigned:
        assigned[key] = EXTRA_COLOURS[len(assigned) % len(EXTRA_COLOURS)]
    return assigned[key]


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def make_plots(rows, out_dir, target):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not installed; wrote tables only", file=sys.stderr)
        return []

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "lines.linewidth": 2,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    made = []
    assigned = {}
    sizes = sorted({r["database_rows"] for r in rows})
    headline = headline_rows(rows, target)

    def series(rows_in):
        grouped = defaultdict(list)
        for r in rows_in:
            key = r["route"] if r["route"] == "ivf" else r["variant"]
            grouped[key].append(r)
        return grouped

    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6.4, 4))
    handles = []
    for key, group in series(headline).items():
        group.sort(key=lambda r: r["database_rows"])
        x = [r["database_rows"] for r in group]
        c = colour_for(key, assigned)
        ax.plot(x, [r["latency_median_ms"] for r in group], marker="o", markersize=6, color=c)
        ax.plot(x, [r["latency_p95_ms"] for r in group], marker="^", markersize=6,
                color=c, linestyle="--", linewidth=1.5)
        label = key + (f" (fastest setting with recall >= {target})" if key == "ivf" else "")
        handles.append(Line2D([], [], color=c, marker="o", label=label))
    handles += [Line2D([], [], color=MUTED, label="solid: median"),
                Line2D([], [], color=MUTED, linestyle="--", linewidth=1.5, label="dashed: p95")]
    style_axes(ax, "Single-thread query latency", "Database vectors", "Latency per query (ms)")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.set_ylim(bottom=0)
    ax.legend(handles=handles, frameon=False, fontsize=8)
    fig.savefig(out_dir / "latency_by_size.png")
    plt.close(fig)
    made.append("latency_by_size.png")

    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.8), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        at_size = [r for r in rows if r["database_rows"] == size]
        ivf = [r for r in at_size if r["route"] == "ivf"]
        for i, lists in enumerate(sorted({r["lists"] for r in ivf})):
            pts = sorted([r for r in ivf if r["lists"] == lists], key=lambda r: r["probes"])
            ax.plot([p["latency_median_ms"] for p in pts], [p["recall_at_k"] for p in pts],
                    marker="o", markersize=6, color=IVF_LIST_COLOURS[i % len(IVF_LIST_COLOURS)],
                    label=f"ivf, {int(lists)} lists (more probes to the right)")
            last = pts[-1]
            ax.annotate(f"{int(lists)} lists", (last["latency_median_ms"], last["recall_at_k"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=8, color=MUTED)
        for r in at_size:
            if r["route"] == "ivf":
                continue
            ax.plot([r["latency_median_ms"]], [r["recall_at_k"]], marker="s", markersize=8,
                    linestyle="none", color=colour_for(r["variant"], assigned), label=r["variant"])
        ax.axhline(target, color=MUTED, linewidth=1, linestyle=":")
        ax.text(ax.get_xlim()[0], target, f" target {target}", va="bottom", fontsize=8, color=MUTED)
        style_axes(ax, f"{size:,} vectors", "Median latency (ms)", "Recall@10")
        ax.set_ylim(min(0.5, min((r["recall_at_k"] or 1) for r in at_size) - 0.05), 1.02)
    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=min(3, len(unique)),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Recall against latency (up and to the left is better)", x=0.01, ha="left",
                 fontweight="bold")
    fig.savefig(out_dir / "recall_vs_latency.png")
    plt.close(fig)
    made.append("recall_vs_latency.png")

    threads = thread_counts(rows)
    biggest = [r for r in headline if r["database_rows"] == sizes[-1]]
    if len(threads) > 1 and biggest:
        fig, ax = plt.subplots(figsize=(5.6, 4))
        ax.plot(threads, threads, color=MUTED, linestyle=":", linewidth=1, label="ideal (linear)")
        for r in biggest:
            ts = [t for t in threads if r.get(f"speedup_t{t}")]
            key = r["route"] if r["route"] == "ivf" else r["variant"]
            ax.plot(ts, [r[f"speedup_t{t}"] for t in ts], marker="o", markersize=7,
                    color=colour_for(key, assigned), label=r["variant"])
        style_axes(ax, f"Batch speedup over one thread ({sizes[-1]:,} vectors)",
                   "Threads", "Speedup (x)")
        ax.set_xticks(threads)
        ax.set_ylim(0, max(threads) * 1.05)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(out_dir / "thread_scaling.png")
        plt.close(fig)
        made.append("thread_scaling.png")

    for metric, scale, title, ylabel, filename in [
            ("peak_memory_bytes", 1 / (1024 * 1024), "Peak resident memory per process",
             "Peak RSS (MiB)", "memory_by_size.png"),
            ("index_build_ms", 1.0, "Index construction time", "Build time (ms)", "build_time.png")]:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for key, group in series(headline).items():
            group.sort(key=lambda r: r["database_rows"])
            values = [r.get(metric) for r in group]
            if any(v is None for v in values):
                continue
            ax.plot([r["database_rows"] for r in group], [v * scale for v in values], marker="o",
                    markersize=6, color=colour_for(key, assigned), label=key)
        style_axes(ax, title, "Database vectors", ylabel)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(out_dir / filename)
        plt.close(fig)
        made.append(filename)
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="default: RUN_DIR/report")
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL_DEFAULT)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    rows = aggregate(load_rows(args.run_dir))
    out_dir = args.output or args.run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tables(rows, out_dir, args.target_recall)
    made = [] if args.no_plots else make_plots(rows, out_dir, args.target_recall)
    print(f"Wrote aggregated.csv, summary.md{''.join(', ' + m for m in made)} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

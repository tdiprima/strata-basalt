#!/usr/bin/env python3
"""Turn results/ (bench_*.json from bench.py, status.tsv and gpu.csv from run_all.sh) into results/REPORT.md.

    python3 report.py results
"""
import csv
import difflib
import glob
import json
import os
import statistics
import sys

BASELINE = "ollama"      # the same-model, same-quantization Ollama run
WORTH_IT = 1.5           # speed-up over the baseline below which switching engines is not worth it
SCORE_SLACK = 0.05       # extraction accuracy may be at most this much below the baseline


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(v, digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}"


def summarize(b):
    runs = [r for r in b["runs"] if "error" not in r]
    return {
        "tps": med(r["gen_tps"] for r in runs),
        "ttft_short": med(r["ttft_s"] for r in runs if not r["prompt"].startswith("long_")),
        "ttft_long": med(r["ttft_s"] for r in runs if r["prompt"].startswith("long_")),
        "score": (lambda s: sum(s) / len(s) if s else None)([r["score"] for r in runs if r.get("score") is not None]),
        "failed": len(b["runs"]) - len(runs),
        "first": {r["prompt"]: r["text"] for r in runs if r["round"] == 0},
    }


def similarity(a, b):
    """Mean text similarity (0 to 1) of the first-round answers two engines gave to the same prompts."""
    both = [p for p in a if p in b]
    return sum(difflib.SequenceMatcher(None, a[p], b[p]).ratio() for p in both) / len(both) if both else None


def recommend(sums, status):
    base, strata = sums.get(BASELINE), sums.get("strata")
    if not strata or not strata["tps"]:
        return f"Strata did not produce numbers on this node ({status.get('strata', ('', 'not run'))[1]}). Stay on Ollama."
    if not base or not base["tps"]:
        return "Strata ran, but there is no Ollama baseline with the same model to compare against. No recommendation."
    ratio = strata["tps"] / base["tps"]
    accurate = strata["score"] is not None and base["score"] is not None and strata["score"] >= base["score"] - SCORE_SLACK
    if ratio < WORTH_IT:
        return (f"Strata generates {ratio:.2f}x as fast as Ollama on the same model. That is below the {WORTH_IT}x "
                "bar for adopting a new engine that upstream has never tested on this hardware. Stay on Ollama.")
    if not accurate:
        return (f"Strata is {ratio:.2f}x as fast as Ollama but its extraction accuracy is lower "
                f"({fmt(strata['score'], 2)} against {fmt(base['score'], 2)}). Not recommended without a quality review.")
    return (f"Strata is {ratio:.2f}x as fast as Ollama on the same model with matching extraction accuracy. "
            "Worth a pilot on non-clinical text. Upstream has not tested this hardware, so pin the commit.")


def main(d):
    benches = {}
    for path in sorted(glob.glob(os.path.join(d, "bench_*.json"))):
        b = json.load(open(path, encoding="utf-8"))
        benches[b["name"]] = b
    status = {}
    if os.path.exists(os.path.join(d, "status.tsv")):
        for row in csv.reader(open(os.path.join(d, "status.tsv"), encoding="utf-8"), delimiter="\t"):
            if len(row) == 3:
                status[row[0]] = (row[1], row[2])   # the last line for an engine wins
    sums = {n: summarize(b) for n, b in benches.items()}
    out = ["# Strata and Basalt on the hopper node: result", ""]
    gpu = os.path.join(d, "gpu.csv")
    if os.path.exists(gpu):
        out += ["## Node", "", "```", open(gpu, encoding="utf-8").read().strip(), "```", ""]
    out += ["## Does it run here?", "", "| Engine | Result | Detail |", "|---|---|---|"]
    out += [f"| {n} | {s} | {why} |" for n, (s, why) in status.items()]
    out += ["", "## Measured", "",
            "Medians over all requests. Same prompts, temperature 0. Memory is system-wide peak while the engine ran.", "",
            "| Engine | Model | Gen tok/s | TTFT short (s) | TTFT long prompt (s) | Load (s) | Peak VRAM (GiB) | Peak RAM (GiB) "
            "| Extraction accuracy | Text similarity to baseline | Failed requests |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, b in benches.items():
        s = sums[n]
        vram = b["vram_peak_mib"] / 1024 if b["vram_peak_mib"] is not None else None
        sim = None if n == BASELINE or BASELINE not in sums else similarity(sums[BASELINE]["first"], s["first"])
        warn = " (GPU not used?)" if vram is not None and vram < 4 else ""
        out.append(f"| {n} | {b['model']} | {fmt(s['tps'])} | {fmt(s['ttft_short'], 2)} | {fmt(s['ttft_long'], 2)} | "
                   f"{fmt(b['load_s'], 0)} | {fmt(vram)}{warn} | {fmt(b['ram_peak_gib'])} | {fmt(s['score'], 2)} | "
                   f"{fmt(sim, 2)} | {s['failed']} |")
    if not benches:
        out.append("| (no engine produced numbers) | | | | | | | | | | |")
    out += ["", "## Recommendation", "", recommend(sums, status), "",
            "Speed claims by the Strata and Basalt authors are self-reported. The numbers above are the only ones "
            "measured on our hardware.", ""]
    text = "\n".join(out)
    with open(os.path.join(d, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")

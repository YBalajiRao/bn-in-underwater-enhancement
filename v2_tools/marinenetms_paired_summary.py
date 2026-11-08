"""MARINENETMS descriptive paired report (Option A): READ-ONLY, CPU-only, no GPU, no training, no checkpoint. For the seeds 0, 1 and 2 it reads ONLY the twelve history.jsonl files
(xf_attrunk_r3_fp32_s*, xfin_attrunk_in_r3_fp32_s*, marinenet_bn_r3_fp32_s*, marinenet_in_r3_fp32_s*) and, for the six MARINE-Net runs, the gate result marinenet_gate.json (to confirm that the gate passed).
It recomputes the exact Val-A metrics from the history files (never from rounded documentation constants) and prints, for each of the four fixed pairs and each metric (primary: last-10-epoch mean PSNR;
secondary: best-epoch PSNR, SSIM at the best epoch, last-10-epoch mean SSIM), the per-seed deltas, and over the three seeds the mean delta, range (max - min), sign pattern and largest absolute delta.
Fixed pairs (marinenetms_config.REPORT_PAIRS; delta = FIRST minus SECOND of the SAME seed, positive = the first is higher):
  norm_effect_in_marinenet = marinenet_in - marinenet_bn      norm_effect_in_attrunk = xfin - xf
  arch_effect_under_bn     = xf - marinenet_bn                arch_effect_under_in   = xfin - marinenet_in
and the per-seed interaction contrast (xfin - xf) - (marinenet_in - marinenet_bn) (positive = the normalisation gain is larger in AT-Trunk than in MARINE-Net).
It applies NO decision rule: no class, no threshold, no margin, no tolerance; it makes no significance, equivalence, generalisation or state-of-the-art claim, and it isolates no single component (MARINE-Net and
AT-Trunk differ in several components at once). It writes NO file (stdout only) and opens no other file; Val-B and UIEB are never referenced. Importing this module has no side effects.
Usage (NOT approved to run until all six MARINE-Net runs exist and have passed their gates): python v2_tools/marinenetms_paired_summary.py [--runs_dir runs_v2] [--json]"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v2_tools import marinenetms_config as mc  # noqa: E402

METRICS = mc.REPORT_METRICS  # ("last10_psnr", "best_psnr", "best_ssim", "last10_ssim"); the first is the primary metric
EPOCHS = mc.EPOCHS
LAST_N = 10
FAMILIES = ("xf", "xfin", "marinenet_bn", "marinenet_in")
NEW_FAMILIES = ("marinenet_bn", "marinenet_in")


def family_run_name(family, seed):
    mc.validate_seed(seed)
    if family == "xf":
        return f"xf_attrunk_r3_fp32_s{seed}"
    if family == "xfin":
        return f"xfin_attrunk_in_r3_fp32_s{seed}"
    if family == "marinenet_bn":
        return mc.run_name("bn", seed)
    if family == "marinenet_in":
        return mc.run_name("in", seed)
    raise ValueError(f"unknown family {family!r}")


def _real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def run_metrics(hist):
    """-> dict of one run's exact Val-A metrics from its history records; raises ValueError unless the history is 100 contiguous epochs (0..99) with finite Val-A PSNR and SSIM. Pure.
    best = the epoch with the maximum Val-A PSNR (the first such epoch on an exact tie); the last-10 means are over epochs 90..99 (math.fsum, exactly rounded)."""
    if not isinstance(hist, list) or [h.get("epoch") if isinstance(h, dict) else None for h in hist] != list(range(EPOCHS)):
        raise ValueError(f"history is not {EPOCHS} contiguous epochs 0..{EPOCHS - 1}")
    try:
        p = [h["val_A"]["psnr_mean"] for h in hist]
        s = [h["val_A"]["ssim_mean"] for h in hist]
    except (KeyError, TypeError) as e:
        raise ValueError(f"history record incomplete ({type(e).__name__}: {e})") from e
    if not all(_real(v) for v in p + s):
        raise ValueError("a Val-A PSNR or SSIM value is missing or not finite")
    b = max(range(EPOCHS), key=lambda i: p[i])
    return {"last10_psnr": math.fsum(p[-LAST_N:]) / LAST_N, "best_psnr": p[b], "best_epoch": b, "best_ssim": s[b], "last10_ssim": math.fsum(s[-LAST_N:]) / LAST_N}


def _sign(d):
    return "+" if d > 0 else ("-" if d < 0 else "0")


def _aggregate(ds):
    return {"deltas": list(ds), "mean": math.fsum(ds) / len(ds), "range": max(ds) - min(ds), "sign_pattern": ",".join(_sign(d) for d in ds), "largest_abs": max(abs(d) for d in ds)}


def paired_report(per_family):
    """per_family = {family: {seed: metrics}} for the four families and seeds 0, 1, 2 (metrics as returned by run_metrics) -> the descriptive report dict. Pure; raises ValueError on a missing family, seed or value.
    Pairs and the contrast are the fixed ones of marinenetms_config. Nothing else is computed."""
    seeds = list(mc.PAIRED_SEEDS)
    if sorted(per_family) != sorted(FAMILIES):
        raise ValueError(f"the report needs exactly the families {list(FAMILIES)}, got {sorted(per_family)}")
    for f in FAMILIES:
        if sorted(per_family[f]) != seeds:
            raise ValueError(f"family {f}: the report needs exactly the seeds {seeds}, got {sorted(per_family[f])}")
        for s in seeds:
            m = per_family[f][s]
            if not isinstance(m, dict) or not all(_real(m.get(k)) for k in METRICS):
                raise ValueError(f"family {f} seed {s}: a metric is missing or not a finite number")
    pairs = {}
    for name, a, b in mc.REPORT_PAIRS:
        per_seed = {s: {k: per_family[a][s][k] - per_family[b][s][k] for k in METRICS} for s in seeds}
        pairs[name] = {"first": a, "second": b, "per_seed": per_seed, "aggregate": {k: _aggregate([per_seed[s][k] for s in seeds]) for k in METRICS}}
    cname, fa, fb = mc.REPORT_CONTRAST
    contrast_per_seed = {s: {k: pairs[fa]["per_seed"][s][k] - pairs[fb]["per_seed"][s][k] for k in METRICS} for s in seeds}
    return {"banner": mc.REPORT_BANNER, "delta_convention": mc.DELTA_CONVENTION, "primary_metric": METRICS[0], "seeds": seeds,
            "families": {f: {s: {k: per_family[f][s][k] for k in (*METRICS, "best_epoch") if k in per_family[f][s]} for s in seeds} for f in FAMILIES},
            "pairs": pairs, "contrast": {"name": cname, "definition": f"{fa} - {fb} (per seed)", "per_seed": contrast_per_seed,
                                         "aggregate": {k: _aggregate([contrast_per_seed[s][k] for s in seeds]) for k in METRICS}}}


def load_history(path):
    """-> the parsed history records of one run (a list of dicts). Reads only that one file; raises ValueError/OSError on a missing or malformed file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gate_passed(path):
    """-> True only if the gate result file is readable JSON with pass == True (and 14 checks, all passed). Reads only that one file."""
    try:
        g = json.load(open(path))
    except (OSError, ValueError):
        return False
    c = g.get("checks") if isinstance(g, dict) else None
    return isinstance(c, dict) and len(c) == 14 and g.get("pass") is True and all(isinstance(v, dict) and v.get("pass") is True for v in c.values())


def collect(runs_dir):
    """-> per_family metrics for the four families and seeds 0, 1, 2 from `runs_dir` (twelve history.jsonl files; the six MARINE-Net marinenet_gate.json files must show a passed gate). Raises ValueError with the reason otherwise."""
    runs_dir = Path(runs_dir)
    out = {f: {} for f in FAMILIES}
    for f in FAMILIES:
        for s in mc.PAIRED_SEEDS:
            name = family_run_name(f, s)
            if f in NEW_FAMILIES and not gate_passed(runs_dir / name / mc.GATE_JSON_FILE):
                raise ValueError(f"{name}/{mc.GATE_JSON_FILE} is missing or does not show a passed 14-check gate; the pair for seed {s} is not reported")
            try:
                out[f][s] = run_metrics(load_history(runs_dir / name / "history.jsonl"))
            except (OSError, ValueError) as e:
                raise ValueError(f"{name}: {type(e).__name__}: {e}") from e
    return out


def format_report(rep):
    """-> the report as plain text (full-precision numbers via repr). Pure."""
    lines = [rep["banner"], "", rep["delta_convention"], f"primary metric: {rep['primary_metric']}", ""]
    for f in FAMILIES:
        lines.append(f"{f}:")
        for s in rep["seeds"]:
            m = rep["families"][f][s]
            lines.append(f"    seed {s}: " + "  ".join(f"{k} {m[k]!r}" for k in METRICS))
    for name, p in rep["pairs"].items():
        lines += ["", f"pair {name} = {p['first']} - {p['second']} (deltas in seed order):"]
        for k in METRICS:
            g = p["aggregate"][k]
            lines.append(f"    {k:<12} deltas {g['deltas']!r}  mean {g['mean']!r}  range {g['range']!r}  signs {g['sign_pattern']}  largest |delta| {g['largest_abs']!r}")
    c = rep["contrast"]
    lines += ["", f"contrast {c['name']} = {c['definition']}:"]
    for k in METRICS:
        g = c["aggregate"][k]
        lines.append(f"    {k:<12} values {g['deltas']!r}  mean {g['mean']!r}  range {g['range']!r}  signs {g['sign_pattern']}  largest |value| {g['largest_abs']!r}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_dir", default=str(ROOT / "runs_v2"))
    ap.add_argument("--json", action="store_true", help="print the report as JSON instead of text")
    a = ap.parse_args()
    try:
        rep = paired_report(collect(a.runs_dir))
    except ValueError as e:
        print(f"STOP: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(rep, indent=1) if a.json else format_report(rep))


if __name__ == "__main__":
    main()

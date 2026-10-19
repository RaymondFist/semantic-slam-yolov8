#!/usr/bin/env python3
"""
aggregate_runs.py -- multi-run ATE aggregation + paired Wilcoxon (n>=10).
========================================================================
For each configuration it computes ATE RMSE (SE3, via compute_ate.py), then
mean +/- std over runs and, for a paired baseline-vs-mask comparison, the
paired Wilcoxon signed-rank test (falls back to a sign test when n<6 or when
all differences are zero). A 95% t-distribution CI is reported per config.

Run files are discovered by glob, so N is whatever the glob matches and an
N>=10 multi-run set can be passed in.

Usage:
    python aggregate_runs.py \
        --gt <gt_file> --format tum \
        --baseline-glob "<dir>/run_*_baseline_*.txt" \
        --mask-glob     "<dir>/run_*_mask_*.txt" \
        [--json out.json]

Environment:
    SEMANTIC_ROOT  repo root, to locate scripts/compute_ate.py (or --compute-ate-dir).
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import scipy.stats as st

EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_compute_ate(explicit):
    cands = []
    if explicit:
        cands.append(explicit)
    root = os.environ.get("SEMANTIC_ROOT")
    if root:
        cands.append(os.path.join(root, "scripts"))
    cands.append(os.path.dirname(EXP_DIR))
    for d in cands:
        if os.path.isfile(os.path.join(d, "compute_ate.py")):
            sys.path.insert(0, d)
            import compute_ate
            return compute_ate
    raise SystemExit("[ERROR] compute_ate.py not found; set SEMANTIC_ROOT or --compute-ate-dir.")


def ates(compute_ate, files, gt):
    vals = []
    for p in sorted(files):
        r = compute_ate.compute_ate(p, gt)
        if r is not None:
            vals.append(r["rmse"])
    return np.array(vals, float)


def t_ci(a, conf=0.95):
    if len(a) < 2:
        return (float("nan"), float("nan"))
    se = a.std(ddof=1) / math.sqrt(len(a))
    tcrit = st.t.ppf(0.5 + conf / 2.0, len(a) - 1)
    return (float(a.mean() - tcrit * se), float(a.mean() + tcrit * se))


def paired_test(base, mask):
    b, m = np.asarray(base, float), np.asarray(mask, float)
    if len(b) != len(m) or len(b) < 1:
        return float("nan"), "n/a"
    d = b - m
    nonz = d[d != 0]
    if len(b) < 6 or len(nonz) == 0:
        tot = len(nonz) if len(nonz) else len(b)
        cnt = int((nonz > 0).sum()) if len(nonz) else tot
        p = st.binomtest(cnt, tot, 0.5, alternative="two-sided").pvalue if tot else float("nan")
        return float(p), "sign"
    try:
        _, p = st.wilcoxon(b, m)
        return float(p), "wilcoxon"
    except ValueError:
        return float("nan"), "tie"


def main():
    ap = argparse.ArgumentParser(description="Aggregate multi-run ATE + paired Wilcoxon")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--format", default="tum", choices=["tum", "kitti"])
    ap.add_argument("--baseline-glob", required=True)
    ap.add_argument("--mask-glob", required=True)
    ap.add_argument("--compute-ate-dir")
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.format == "kitti":
        raise SystemExit("[ERROR] kitti aggregation needs 12-col handling; use traj_eval.py per run "
                         "and aggregate the resulting .json values, or convert to TUM format first.")
    if not os.path.isfile(args.gt):
        raise SystemExit(f"[ERROR] gt not found: {args.gt}")

    compute_ate = load_compute_ate(args.compute_ate_dir)
    b_files = glob.glob(args.baseline_glob)
    m_files = glob.glob(args.mask_glob)
    if not b_files or not m_files:
        raise SystemExit(f"[ERROR] glob matched no files:\n  baseline: {args.baseline_glob} -> {len(b_files)}\n"
                         f"  mask: {args.mask_glob} -> {len(m_files)}")

    b = ates(compute_ate, b_files, args.gt)
    m = ates(compute_ate, m_files, args.gt)
    if len(b) == 0 or len(m) == 0:
        raise SystemExit("[ERROR] no valid ATE values (check trajectories vs gt).")

    out = {"n_baseline": len(b), "n_mask": len(m),
           "baseline": {"mean": float(b.mean()),
                        "std": float(b.std(ddof=1)) if len(b) > 1 else None,
                        "ci95": t_ci(b), "values": [float(x) for x in b]},
           "mask": {"mean": float(m.mean()),
                    "std": float(m.std(ddof=1)) if len(m) > 1 else None,
                    "ci95": t_ci(m), "values": [float(x) for x in m]}}

    print(f"baseline  n={len(b)}  mean={b.mean():.4f} "
          f"+/- {(b.std(ddof=1) if len(b) > 1 else 0):.4f}")
    print(f"mask      n={len(m)}  mean={m.mean():.4f} "
          f"+/- {(m.std(ddof=1) if len(m) > 1 else 0):.4f}")

    paired = False
    if len(b) == len(m):
        p, name = paired_test(b, m)
        delta = (m.mean() - b.mean()) / b.mean() * 100.0 if b.mean() else float("nan")
        out["paired"] = {"delta_pct": float(delta), "p": p, "test": name}
        print(f"paired delta (mask vs baseline) = {delta:+.2f}%   p={p} ({name})  n={len(b)}")
        paired = True
    else:
        print(f"[WARN] baseline n={len(b)} != mask n={len(m)}; paired test skipped "
              f"(run the same number of runs for each config).")

    if len(b) < 10:
        print(f"[NOTE] n={len(b)} < 10: the two-sided minimum attainable Wilcoxon p "
              f"is above 0.05 at small n; increase N (study requires N>=10).")
    if paired and all((np.asarray(b) - np.asarray(m)) == 0):
        print("[NOTE] all paired differences are exactly zero -> runs were deterministic; "
              "use run_multiseed.sh with a per-run settings variation to obtain spread.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()

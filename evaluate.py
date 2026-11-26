#!/usr/bin/env python3
"""
Measure Package Watcher detection accuracy and alert speed.

1. Label photos at http://<pi-tailscale-ip>:8080/label (Package / No package / Unclear).
2. Run on the Pi:
     python3 evaluate.py
     python3 evaluate.py --moondream 20   # also re-time Moondream on 20 photos
3. Read eval/results.md.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import package_watcher as pw  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
NOT_JUDGED = ("Snoozed", "Alert cooldown")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def scores(pairs):
    tp = sum(1 for t, p in pairs if t and p)
    fp = sum(1 for t, p in pairs if not t and p)
    fn = sum(1 for t, p in pairs if t and not p)
    tn = sum(1 for t, p in pairs if not t and not p)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
            "accuracy": (tp + tn) / len(pairs) if pairs else None,
            "recall_ci": wilson(tp, tp + fn), "precision_ci": wilson(tp, tp + fp)}


def pct(x):
    return "–" if x is None else f"{x * 100:.0f}%"


def secs(x):
    return "–" if x is None else f"{x:.1f} s" if x >= 1 else f"{x * 1000:.0f} ms"


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def latency(values):
    return {"n": len(values), "median": percentile(values, 0.5), "p90": percentile(values, 0.9),
            "max": max(values) if values else None}


def device_name():
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return "unknown device"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moondream", type=int, default=0, metavar="N",
                    help="re-run Moondream on N labeled photos to time it")
    args = ap.parse_args()

    labels = pw.load_labels()
    events = {e["id"]: e for e in pw.db_all() if e.get("mode") == "auto"}
    labeled = [(events[i], v == "package") for i, v in labels.items() if i in events and v in ("package", "none")]
    unclear = sum(1 for i, v in labels.items() if i in events and v == "unclear")
    if not labeled:
        sys.exit("No labeled photos yet. Label some at http://<pi-tailscale-ip>:8080/label first.")

    results = {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "device": device_name(),
        "labeled": len(labeled),
        "with_package": sum(1 for _, t in labeled if t),
        "unclear_skipped": unclear,
        "methods": {},
    }

    judged = [(e, t) for e, t in labeled if not str(e.get("reason", "")).startswith(NOT_JUDGED)]
    results["methods"]["Package Watcher (Moondream)"] = scores(
        [(t, bool(e.get("package_present"))) for e, t in judged]
    )

    moon_s = []
    if args.moondream:
        pick = labeled[: args.moondream]
        pairs = []
        for k, (e, t) in enumerate(pick, 1):
            path = os.path.join(pw.EVENTS_DIR, e.get("image", ""))
            if not os.path.exists(path):
                continue
            started = time.perf_counter()
            try:
                raw = pw.describe_image(path, pw.PACKAGE_PROMPT)
                present, _ = pw.parse_package_response(raw)
            except Exception as ex:
                print(f"\nMoondream failed: {ex}")
                break
            moon_s.append(time.perf_counter() - started)
            pairs.append((t, present))
            print(f"\rMoondream: {k}/{len(pick)} photos ({moon_s[-1]:.1f} s)", end="", flush=True)
        print()
        if pairs:
            results["methods"]["Moondream (re-run now)"] = scores(pairs)

    timed = [e["timing"] for e in events.values() if e.get("timing")]
    stages = [("described", "Motion → Moondream check finished"),
              ("alert", "Motion → phone notification sent"),
              ("total", "Motion → event fully processed")]
    results["latency"] = {key: {"label": label, **latency([t[key] for t in timed if key in t])}
                          for key, label in stages}
    if moon_s:
        results["latency"]["moondream_rerun"] = {"label": "Moondream, re-run now", **latency(moon_s)}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    report = render(results)
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write(report)
    print("\n" + report)


def render(r):
    lines = [
        "# Package Watcher evaluation", "",
        f"Generated {r['generated']} on {r['device']}. {r['labeled']} labeled photos "
        f"({r['with_package']} with a package, {r['labeled'] - r['with_package']} without); "
        f"{r['unclear_skipped']} unclear skipped.", "",
        "| Method | Precision | Recall | F1 | False alarm rate | Photos |", "|---|---|---|---|---|---|",
    ]
    for name, s in r["methods"].items():
        lines.append(f"| {name} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} | "
                     f"{pct(s['false_alarm_rate'])} | {s['n']} |")
    lines += ["", "## Speed", "", "| Stage | Median | 90th percentile | Max | Events |", "|---|---|---|---|---|"]
    for v in r["latency"].values():
        if v["n"]:
            lines.append(f"| {v['label']} | {secs(v['median'])} | {secs(v['p90'])} | {secs(v['max'])} | {v['n']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

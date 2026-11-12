#!/usr/bin/env python3
"""
Measure how well Room Guardian works: detection accuracy and alert speed.

1. Label event photos in the dashboard: http://<pi-tailscale-ip>:8080/label
   (Person / No person / Unclear; the AI's answers are hidden so they can't bias you.)
2. Run on the Pi:
     python3 evaluate.py                  # accuracy + latency report
     python3 evaluate.py --moondream 20   # also re-time Moondream on 20 photos (slow)
3. Read eval/results.md (numbers only, no photos: safe to commit).

What is measured
- Accuracy against your labels, for each detector and for the whole system's loud-alert decision.
- YOLO11n is re-run on every saved photo at several thresholds (precision/recall trade-off).
- Latency from the timing stored with each event (events recorded after this update).
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import room_guardian as rg  # noqa: E402  (reuses the same detector, patterns and paths)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
NOT_JUDGED = ("Snoozed", "Alert already sent")  # events skipped by the gate, not by detection


# --- Metrics ---
def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion (honest error bars on small samples)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def scores(pairs):
    """pairs: [(truth, predicted)] booleans -> confusion counts and rates."""
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


# --- Predictions ---
def moondream_says_person(text):
    text = (text or "").lower()
    if not text or text.startswith("(vision failed"):
        return None
    return bool(rg.PERSON_PATTERN.search(text)) and not rg.NEGATED_PATTERN.search(text)


def system_loud(e):
    """Did the phone get a loud alert? (person or suspicious, from any detector)"""
    return bool(e.get("alert") or (e.get("labels") or {}).get("person") or e.get("faces", 0) > 0)


def yolo_score(net, frame):
    """Highest person confidence in the frame (threshold applied later)."""
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (rg.PERSON_INPUT, rg.PERSON_INPUT), swapRB=True)
    net.setInput(blob)
    out = net.forward()[0].T
    return float(out[:, 4].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moondream", type=int, default=0, metavar="N",
                    help="also re-run Moondream on N labeled photos to time it (about 10-30 s each)")
    args = ap.parse_args()

    labels = rg.load_labels()
    events = {e["id"]: e for e in rg.db_all() if e.get("mode") == "auto"}
    labeled = [(events[i], v == "person") for i, v in labels.items() if i in events and v in ("person", "none")]
    unclear = sum(1 for i, v in labels.items() if i in events and v == "unclear")
    if not labeled:
        sys.exit("No labeled photos yet. Label some at http://<pi-tailscale-ip>:8080/label first.")
    people = sum(1 for _, t in labeled if t)
    print(f"{len(labeled)} labeled photos ({people} with a person, {len(labeled) - people} without), "
          f"{unclear} unclear skipped")

    results = {"generated": datetime.now().isoformat(timespec="minutes"), "device": device_name(),
               "labeled": len(labeled), "with_person": people, "unclear_skipped": unclear, "methods": {}}

    # Decisions made at capture time (stored with each event)
    judged = [(e, t) for e, t in labeled if not str(e.get("reason", "")).startswith(NOT_JUDGED)]
    results["methods"]["Full system (loud alert)"] = scores([(t, system_loud(e)) for e, t in judged])
    md = [(t, moondream_says_person(e.get("description"))) for e, t in labeled]
    results["methods"]["Moondream description"] = scores([(t, p) for t, p in md if p is not None])
    results["methods"]["Face detector (Haar)"] = scores([(t, e.get("faces", 0) > 0) for e, t in labeled])

    # YOLO11n re-run on the saved photos
    sweep, yolo_ms = [], []
    if os.path.exists(rg.PERSON_MODEL_PATH):
        net = cv2.dnn.readNetFromONNX(rg.PERSON_MODEL_PATH)
        yolo_score(net, np.zeros((480, 640, 3), np.uint8))  # warm up
        scored = []
        for k, (e, t) in enumerate(labeled, 1):
            frame = cv2.imread(os.path.join(rg.EVENTS_DIR, e.get("image", "")))
            if frame is None:
                continue  # photo already deleted (only the newest MAX_IMAGES are kept)
            started = time.perf_counter()
            scored.append((t, yolo_score(net, frame)))
            yolo_ms.append(time.perf_counter() - started)
            print(f"\rYOLO11n: {k}/{len(labeled)} photos", end="", flush=True)
        print()
        for th in THRESHOLDS:
            sweep.append({"threshold": th, **scores([(t, s >= th) for t, s in scored])})
        results["methods"][f"YOLO11n (threshold {rg.PERSON_CONF:.2f})"] = scores(
            [(t, s >= rg.PERSON_CONF) for t, s in scored])
    else:
        print(f"YOLO model not found at {rg.PERSON_MODEL_PATH}: skipping the detector re-run")
    results["yolo_sweep"] = sweep

    # Optional: time Moondream directly
    moon_s = []
    if args.moondream:
        people_first = sorted(labeled, key=lambda et: not et[1])
        pick = people_first[: args.moondream // 2] + people_first[::-1][: args.moondream - args.moondream // 2]
        pairs = []
        for k, (e, t) in enumerate(pick, 1):
            path = os.path.join(rg.EVENTS_DIR, e.get("image", ""))
            if not os.path.exists(path):
                continue
            started = time.perf_counter()
            try:
                text = rg.describe_image(path, rg.DESCRIBE_PROMPT)
            except Exception as ex:
                print(f"\nMoondream failed: {ex}")
                break
            moon_s.append(time.perf_counter() - started)
            pairs.append((t, moondream_says_person(text)))
            print(f"\rMoondream: {k}/{len(pick)} photos ({moon_s[-1]:.1f} s)", end="", flush=True)
        print()
        if pairs:
            results["methods"]["Moondream (re-run now)"] = scores([(t, p) for t, p in pairs if p is not None])

    # Latency (from timing stored with each event)
    timed = [e["timing"] for e in events.values() if e.get("timing")]
    stages = [("detect", "Motion → best frame chosen (burst + YOLO)"),
              ("alert", "Motion → first phone notification sent"),
              ("describe_s", "Moondream description (time to run)"),
              ("followup", "Motion → follow-up with description"),
              ("total", "Motion → event fully processed")]
    results["latency"] = {key: {"label": label, **latency([t[key] for t in timed if key in t])}
                          for key, label in stages}
    results["latency"]["yolo_inference"] = {"label": "YOLO11n inference per photo", **latency(yolo_ms)}
    if moon_s:
        results["latency"]["moondream_rerun"] = {"label": "Moondream, re-run now", **latency(moon_s)}

    # Alert noise per day (from labeled days)
    days = {rg._local(e).date() for e, _ in labeled}
    sysm = results["methods"]["Full system (loud alert)"]
    results["per_day"] = {"days": len(days), "false_loud_alerts": sysm["fp"] / len(days) if days else None,
                          "missed_people": sysm["fn"] / len(days) if days else None}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    report = render(results)
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved {os.path.join(OUT_DIR, 'results.md')} and results.json")


def device_name():
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return "unknown device"


def render(r):
    sysm = r["methods"]["Full system (loud alert)"]
    lat = r["latency"]
    lines = [
        "# Room Guardian evaluation", "",
        f"Generated {r['generated']} on {r['device']}. {r['labeled']} hand-labeled event photos "
        f"({r['with_person']} with a person, {r['labeled'] - r['with_person']} without); "
        f"{r['unclear_skipped']} marked unclear were left out.", "",
        "## Headline", "",
    ]
    if sysm["recall"] is not None:
        lo, hi = sysm["recall_ci"]
        lines.append(f"- **People caught:** {pct(sysm['recall'])} ({sysm['tp']}/{sysm['tp'] + sysm['fn']}; "
                     f"95% CI {pct(lo)}–{pct(hi)})")
    if sysm["false_alarm_rate"] is not None:
        lines.append(f"- **False loud alerts:** {pct(sysm['false_alarm_rate'])} of empty-room motion events "
                     f"({sysm['fp']}/{sysm['fp'] + sysm['tn']})")
    if lat["alert"]["n"]:
        lines.append(f"- **Motion → phone notification:** median {secs(lat['alert']['median'])}, "
                     f"90th percentile {secs(lat['alert']['p90'])} ({lat['alert']['n']} events)")
    if lat["followup"]["n"]:
        lines.append(f"- **Description arrives:** median {secs(lat['followup']['median'])} after motion")
    lines += ["", "## Detection accuracy", "",
              "| Method | Precision | Recall | F1 | False alarm rate | Photos |", "|---|---|---|---|---|---|"]
    for name, m in r["methods"].items():
        lines.append(f"| {name} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | "
                     f"{pct(m['false_alarm_rate'])} | {m['n']} |")
    if r["yolo_sweep"]:
        lines += ["", "## YOLO11n confidence threshold", "",
                  "| Threshold | Precision | Recall | F1 | False alarm rate |", "|---|---|---|---|---|"]
        best = max(r["yolo_sweep"], key=lambda s: s["f1"] or 0)
        best = best if best["f1"] else None
        for s in r["yolo_sweep"]:
            mark = " ← best F1" if s is best else ""
            lines.append(f"| {s['threshold']:.2f}{mark} | {pct(s['precision'])} | {pct(s['recall'])} | "
                         f"{pct(s['f1'])} | {pct(s['false_alarm_rate'])} |")
    lines += ["", "## Speed", "", "| Stage | Median | 90th percentile | Max | Events |", "|---|---|---|---|---|"]
    for v in lat.values():
        if v["n"]:
            lines.append(f"| {v['label']} | {secs(v['median'])} | {secs(v['p90'])} | {secs(v['max'])} | {v['n']} |")
    pd = r["per_day"]
    if pd["days"]:
        lines += ["", f"Over {pd['days']} labeled day(s): {pd['false_loud_alerts']:.1f} false loud alerts "
                      f"and {pd['missed_people']:.1f} missed people per day."]
    lines += ["", "## How this was measured", "",
              "- Ground truth: each motion-event photo labeled by hand (person visible or not) in the "
              "dashboard's /label page, with the AI's answers hidden.",
              "- *Full system* is the decision that matters: did the phone get a loud alert (any detector "
              "finding a person, or something suspicious). Events skipped by snooze or the alert cooldown "
              "are excluded.",
              "- Recall only counts people who triggered motion detection; someone who never caused motion "
              "isn't in the data.",
              "- Latency is measured on the Pi from motion detection to ntfy accepting the notification; "
              "delivery to the phone adds network time (usually well under a second).",
              "- Small samples are noisy: the 95% confidence interval shows the realistic range.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

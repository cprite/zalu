"""Evaluation core (owner: Damir): pure functions, no GPU, no files. Everything here is unit-tested.

evaluate_model()   ground truth + predictions -> one per-model entry of EvaluationReport.per_model
compare()          baseline vs fine-tuned deltas, synthetic A/B verdict
readiness()        deployment recommendation against explicit acceptance criteria
select_threshold() pick a confidence threshold on the VALIDATION split (never on the test split)
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from backend.evaluation_engine import metrics as M
from backend.evaluation_engine.matching import Box, iou, match_image

DEFAULT_IOU = 0.5
DEFAULT_CONF = 0.25
MAX_FAILURES = 30  # failure gallery entries kept per model (counts are always complete)


@dataclass(frozen=True)
class Criteria:
    """Acceptance criteria for "ready to deploy". Shown in the Report Card next to the measured values."""
    min_recall: float = 0.80
    min_precision: float = 0.70
    min_fps: Optional[float] = 10.0
    min_gt_per_class: int = M.MIN_GT_PER_CLASS
    min_images: int = M.MIN_IMAGES

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def to_boxes(items: Iterable, default_image: Optional[str] = None) -> list[Box]:
    """Accept schemas.Annotation / schemas.Detection / dicts / Box and return Box objects."""
    out = []
    for it in items:
        if isinstance(it, Box):
            out.append(it)
            continue
        d = it if isinstance(it, dict) else it.model_dump()
        out.append(Box.of(d.get("image") or default_image, d["cls"], d["box"], d.get("score")))
    return out


def evaluate_model(
    gt: list[Box],
    preds: list[Box],
    images: list[str],
    classes: list[str],
    iou_threshold: float = DEFAULT_IOU,
    conf_threshold: float = DEFAULT_CONF,
    sessions: Optional[dict[str, str]] = None,
    latencies_ms: Optional[list[float]] = None,
    criteria: Criteria = Criteria(),
    n_sessions: Optional[int] = None,
) -> dict:
    """Evaluate one model on one test set. `images` lists every test image, including ones with no objects
    (they matter: every prediction on an empty frame is a false positive)."""
    img_set = set(images)
    unknown = {b.image for b in gt} - img_set
    if unknown:
        raise ValueError(f"ground truth refers to images not in the test list: {sorted(unknown)[:3]}")
    cls_set = set(classes)
    gt = [b for b in gt if b.cls in cls_set]
    preds = [p for p in preds if p.cls in cls_set and p.image in img_set]

    by_img_gt: dict[str, list[Box]] = defaultdict(list)
    by_img_pr: dict[str, list[Box]] = defaultdict(list)
    for b in gt:
        by_img_gt[b.image].append(b)
    for p in preds:
        by_img_pr[p.image].append(p)

    cnt = {c: Counter() for c in classes}
    img_cnt = {c: Counter() for c in classes}
    fp_reasons, fn_reasons = Counter(), Counter()
    failures: list[dict] = []

    for im in images:
        o = match_image(by_img_gt[im], by_img_pr[im], iou_threshold, conf_threshold)
        for p, _, _ in o.tp:
            cnt[p.cls]["tp"] += 1
        for p, reason, best in o.fp:
            cnt[p.cls]["fp"] += 1
            fp_reasons[reason] += 1
            failures.append({"image": im, "kind": "false_positive", "reason": reason, "cls": p.cls,
                             "box": list(p.box), "score": p.score, "iou": round(best, 3)})
        for g, reason, near in o.fn:
            cnt[g.cls]["fn"] += 1
            fn_reasons[reason] += 1
            failures.append({"image": im, "kind": "false_negative", "reason": reason, "cls": g.cls,
                             "box": list(g.box), "score": near.score if near else None,
                             "predicted_cls": near.cls if near else None,
                             "predicted_box": list(near.box) if near else None,
                             "iou": round(iou(near.box, g.box), 3) if near else 0.0})
        # image level: "did the frame contain class c" vs "did the model flag class c"
        for c in classes:
            has = any(b.cls == c for b in by_img_gt[im])
            flagged = any(p.cls == c and (p.score or 0) >= conf_threshold for p in by_img_pr[im])
            img_cnt[c]["tp" if has and flagged else "fp" if flagged else "fn" if has else "tn"] += 1

    tot = sum((cnt[c] for c in classes), Counter())
    overall = M.prf(tot["tp"], tot["fp"], tot["fn"])
    gt_per_class = {c: cnt[c]["tp"] + cnt[c]["fn"] for c in classes}
    if sessions:
        n_sessions = len(set(sessions.values()))

    lat = sorted(latencies_ms or [])
    perf = None
    if lat:
        med = statistics.median(lat)
        perf = {"latency_ms_median": round(med, 1), "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
                "fps": round(1000 / med, 1) if med > 0 else None, "n": len(lat),
                "note": "model time on the backend GPU, excludes network and decoding"}

    warns = M.sample_warnings(len(images), gt_per_class, n_sessions, overall)
    failures.sort(key=lambda f: (f["kind"] != "false_negative", -(f["score"] or 0)))
    entry = {
        "settings": {"iou_threshold": iou_threshold, "conf_threshold": conf_threshold,
                     "matching": "greedy per class by descending score, one prediction per object"},
        "test": {"n_images": len(images), "n_images_without_objects": sum(1 for i in images if not by_img_gt[i]),
                 "n_sessions": n_sessions, "gt_per_class": gt_per_class, "classes": list(classes)},
        "overall": overall,
        "per_class": {c: {**M.prf(cnt[c]["tp"], cnt[c]["fp"], cnt[c]["fn"]),
                          "image_level": {**M.prf(img_cnt[c]["tp"], img_cnt[c]["fp"], img_cnt[c]["fn"]),
                                          "tn": img_cnt[c]["tn"]}}
                      for c in classes},
        "failure_counts": {"false_positive": dict(fp_reasons), "false_negative": dict(fn_reasons)},
        "failures": failures[:MAX_FAILURES],
        "failures_total": len(failures),
        "performance": perf,
        "warnings": warns,
    }
    entry["readiness"] = readiness(entry, criteria)
    # flat copies for simple consumers (schemas.EvaluationReport documents precision/recall/f1/fp/fn/fps)
    entry.update({k: overall[k] for k in ("precision", "recall", "f1", "fp", "fn")})
    entry["fps"] = perf["fps"] if perf else None
    return entry


def readiness(entry: dict, c: Criteria) -> dict:
    o, t, perf = entry["overall"], entry["test"], entry.get("performance")
    checks = [
        _check("Recall (share of real objects found)", ">=", c.min_recall, o["recall"]),
        _check("Precision (share of alarms that are real)", ">=", c.min_precision, o["precision"]),
    ]
    if c.min_fps is not None:
        checks.append(_check("Speed, frames per second", ">=", c.min_fps, perf["fps"] if perf else None))
    data_checks = [_check("Test images", ">=", c.min_images, t["n_images"])]
    data_checks += [_check(f"Test '{k}' objects", ">=", c.min_gt_per_class, n) for k, n in t["gt_per_class"].items()]

    metric_ok = all(x["passed"] for x in checks)
    data_ok = all(x["passed"] for x in data_checks)
    if not metric_ok:
        failed = [x["name"] for x in checks if not x["passed"]]
        verdict, summary = "not_ready", "Does not meet: " + "; ".join(failed) + "."
    elif not data_ok:
        verdict = "insufficient_data"
        summary = "Meets the targets on this test set, but the test set is too small to confirm it. Capture more independent footage."
    else:
        verdict, summary = "ready", "Meets all acceptance criteria on independent test footage."
    lo = (o.get("recall_ci95") or [None])[0]
    if verdict == "ready" and lo is not None and lo < c.min_recall:
        summary += f" Note: the lower end of the recall interval ({lo:.2f}) is below the target."
    summary += " A small test set never proves production safety; monitor the model after deployment."
    return {"verdict": verdict, "summary": summary, "criteria": checks, "data_criteria": data_checks,
            "criteria_config": c.as_dict()}


def _check(name: str, op: str, target: float, value) -> dict:
    passed = value is not None and value >= target
    return {"name": name, "op": op, "target": target, "value": value, "passed": passed}


def compare(per_model: dict[str, dict], meta: dict[str, dict], baseline_id: Optional[str] = None) -> dict:
    """Deltas against a baseline (default: the zero-shot model, else the first) and the synthetic A/B verdict.

    meta: model_id -> {"kind": "trained"|"zero_shot", "variant": ...}. A difference is called "clear" only when
    the 95% intervals do not overlap, a deliberately conservative rule for small test sets.
    """
    ids = list(per_model)
    if not ids:
        return {}
    if baseline_id is None:
        baseline_id = next((i for i in ids if meta.get(i, {}).get("kind") == "zero_shot"), ids[0])
    base = per_model[baseline_id]["overall"]
    out = {"baseline_id": baseline_id, "deltas": {}, "notes": []}
    for i in ids:
        if i == baseline_id:
            continue
        o = per_model[i]["overall"]
        d = {k: _delta(o[k], base[k]) for k in ("precision", "recall", "f1")}
        d["fp"], d["fn"] = o["fp"] - base["fp"], o["fn"] - base["fn"]
        d["recall_clear"] = _separated(o["recall_ci95"], base["recall_ci95"])
        d["precision_clear"] = _separated(o["precision_ci95"], base["precision_ci95"])
        out["deltas"][i] = d
        per_model[i]["delta_vs_baseline"] = {"baseline_id": baseline_id, **d}

    real = [i for i in ids if meta.get(i, {}).get("kind") == "trained" and meta[i].get("variant") == "real_only"]
    syn = [i for i in ids if meta.get(i, {}).get("kind") == "trained" and meta[i].get("variant") == "real_plus_synthetic"]
    if real and syn:
        r, s = per_model[real[0]]["overall"], per_model[syn[0]]["overall"]
        if r["f1"] is None or s["f1"] is None:
            verdict = "unknown"
        elif s["f1"] > r["f1"] and _separated(s["recall_ci95"], r["recall_ci95"]):
            verdict = "helped"
        elif s["f1"] > r["f1"]:
            verdict = "slightly_better_not_clear"
        else:
            verdict = "did_not_help"
        out["synthetic_ab"] = {"real_only": real[0], "real_plus_synthetic": syn[0], "verdict": verdict,
                               "f1_real_only": r["f1"], "f1_real_plus_synthetic": s["f1"]}
        msg = {"helped": "Synthetic data clearly improved results on real test footage.",
               "slightly_better_not_clear": "Synthetic data gave a small gain on real test footage, within noise.",
               "did_not_help": "Synthetic data did NOT improve results on real test footage.",
               "unknown": "Synthetic A/B could not be measured."}[verdict]
        out["notes"].append(f"{msg} (F1 real-only {r['f1']}, real+synthetic {s['f1']})")
    return out


def select_threshold(
    gt: list[Box], preds: list[Box], images: list[str], classes: list[str],
    iou_threshold: float = DEFAULT_IOU, candidates: Optional[list[float]] = None,
) -> dict:
    """Choose the confidence threshold with the best F1 on the VALIDATION split. Then freeze it for testing."""
    candidates = candidates or [round(0.05 * k, 2) for k in range(1, 19)]
    curve = []
    for t in candidates:
        e = evaluate_model(gt, preds, images, classes, iou_threshold, t)
        curve.append({"threshold": t, "precision": e["precision"], "recall": e["recall"], "f1": e["f1"]})
    best = max(curve, key=lambda r: (r["f1"] or -1, r["threshold"]))
    return {"threshold": best["threshold"], "f1": best["f1"], "curve": curve,
            "note": "selected on the validation split; do not re-tune on the test split"}


def _delta(a, b):
    return None if a is None or b is None else round(a - b, 4)


def _separated(a, b) -> bool:
    return bool(a and b and (a[0] > b[1] or b[0] > a[1]))

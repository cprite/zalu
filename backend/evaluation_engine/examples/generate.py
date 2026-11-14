"""Regenerate the synthetic report and comparison examples from local JSON."""

import json
from pathlib import Path

from backend.evaluation_engine import (
    EvaluationConfig,
    ReadinessThresholds,
    build_evaluation_report,
    compare_models,
    evaluate_detections,
    load_json,
)


DIRECTORY = Path(__file__).resolve().parent


def save(name, data):
    path = DIRECTORY / f"{name}.synthetic.json"
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    truth = load_json(DIRECTORY / "ground_truth.synthetic.json")
    config = EvaluationConfig(confidence_threshold=0.5)
    readiness = ReadinessThresholds(0.8, 0.8, 0.8, 4, 4, {"person": 2, "vest": 2})
    reports = {}
    for variant in ("pretrained_baseline", "fine_tuned", "real_only", "real_plus_synthetic"):
        predictions = load_json(DIRECTORY / f"{variant}.synthetic.json")
        results = evaluate_detections(truth, predictions, config)
        reports[variant] = build_evaluation_report(results, readiness=readiness)
    save("baseline_report", reports["pretrained_baseline"])
    save("fine_tuned_report", reports["fine_tuned"])
    save("baseline_vs_fine_tuned", compare_models(reports["pretrained_baseline"], reports["fine_tuned"]))
    save("real_vs_real_plus_synthetic", compare_models(reports["real_only"], reports["real_plus_synthetic"]))


if __name__ == "__main__":
    main()

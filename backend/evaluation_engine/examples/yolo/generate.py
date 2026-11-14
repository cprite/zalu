"""Reproduce the synthetic FastAPI response example without GPU dependencies."""

import json
from pathlib import Path

from backend.evaluation_engine import EvaluationConfig, ReadinessThresholds, load_json
from backend.evaluation_engine.yolo import (
    YoloLabelImage, evaluate_yolo, from_inference_result, load_yolo_ground_truth,
)


DIRECTORY = Path(__file__).resolve().parent
OUTPUT = DIRECTORY / "api_report.synthetic.json"


def build_example():
    metadata = load_json(DIRECTORY / "roster.synthetic.json")
    roster = [
        YoloLabelImage(**{**row, "label_path": DIRECTORY / row["label_path"]})
        for row in metadata.pop("images")
    ]
    truth = load_yolo_ground_truth(roster, **metadata)
    predictions = load_json(DIRECTORY / "inference.synthetic.json")
    per_model = {}
    for key, variant, class_map in (
        ("baseline", "pretrained_baseline", {"worker": "person"}),
        ("trained", "real_only", {}),
    ):
        packets = predictions[key]
        model_id = packets[roster[0].image_id]["model_id"]
        per_model[model_id] = evaluate_yolo(
            truth, (from_inference_result(frame_id, packet, class_map=class_map)
                    for frame_id, packet in packets.items()),
            EvaluationConfig(0.5, 0.5), model_id=model_id, variant=variant,
            readiness=ReadinessThresholds(0.8, 0.8, 0.8, 2, 2, {"vest": 1}),
        )
    # The example deliberately uses the legacy envelope's mock flag.
    return {
        "id": "rep_synthetic_yolo",
        "request": {"model_ids": list(per_model), "test_dataset_id": metadata["dataset_id"]},
        "per_model": per_model,
        "notes": ["Synthetic adapter fixture. Scores are illustrative only."],
        "mock": True,
        "created_at": 0.0,
    }


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build_example(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

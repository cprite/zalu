"""Data Engine CLI.

  python -m backend.data_engine run --task spec.json --media clip1.mp4 clip2.mp4 --out runs/vest
  python -m backend.data_engine ingest   --media clip.mp4 --out work/
  python -m backend.data_engine annotate --task spec.json --frames work/frames.json --out work/annotations.json
  python -m backend.data_engine correct  --annotations work/annotations.json --corrections fixes.json
  python -m backend.data_engine export   --task spec.json --annotations work/annotations.json --frames work/frames.json --out ds/
  python -m backend.data_engine export-test --task spec.json --annotations a.json --sessions s_test_1 --out test_ds/
  python -m backend.data_engine validate ds/
  python -m backend.data_engine synthesize --task spec.json --frames work/frames.json --annotations work/annotations.json \
      --exclude-sessions <val/test sessions> --max-images 20 --out work/synthetic   (needs FAL_KEY)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from backend.task_engine import TaskSpec, compile_task

from . import (AnnotationSet, FrameCollection, LabelCorrection, apply_corrections, auto_annotate, export_dataset,
               export_test_set, process_media, validate_dataset)
from .preview import render_previews


def _load_task(args) -> TaskSpec:
    if args.task:
        return TaskSpec.model_validate_json(Path(args.task).read_text())
    spec = compile_task(args.prompt)
    if spec.status != "ok":
        sys.exit(f"TaskSpec status={spec.status}: {spec.issues} {spec.clarification_questions}")
    return spec


def _sessions(pairs):
    out = {}
    for p in pairs or []:
        f, _, s = p.partition("=")
        out[f] = s
    return out


def _detector(args):
    from .annotate import YoloWorldDetector
    return YoloWorldDetector(weights=args.weights, device=args.device)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # fal polls its queue many times per request
    ap = argparse.ArgumentParser(prog="python -m backend.data_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def task_args(p):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--task", help="TaskSpec JSON file")
        g.add_argument("--prompt", help="Natural-language task (calls Claude)")

    def ingest_args(p):
        p.add_argument("--media", nargs="+", required=True, help="Videos, images or folders")
        p.add_argument("--session", action="append", metavar="FILE=SESSION_ID",
                       help="Assign a file to a recording session (repeatable)")
        p.add_argument("--fps", type=float, default=1.0)
        p.add_argument("--max-frames", type=int, default=200)
        p.add_argument("--min-sharpness", type=float, default=25.0)

    def det_args(p):
        p.add_argument("--weights", default="yolov8s-worldv2.pt")
        p.add_argument("--device", default=None, help="cpu, mps, 0 ...")

    p = sub.add_parser("ingest"); ingest_args(p); p.add_argument("--out", required=True)
    p = sub.add_parser("annotate"); task_args(p); det_args(p)
    p.add_argument("--frames", required=True); p.add_argument("--out", required=True)
    p.add_argument("--preview", help="Write annotated preview JPEGs here")
    p = sub.add_parser("correct")
    p.add_argument("--annotations", required=True); p.add_argument("--corrections", required=True)
    p.add_argument("--out", help="Default: overwrite --annotations")
    p = sub.add_parser("export"); task_args(p)
    p.add_argument("--annotations", required=True); p.add_argument("--frames")
    p.add_argument("--out", required=True); p.add_argument("--val-sessions", nargs="*")
    p.add_argument("--held-out", nargs="*", default=[]); p.add_argument("--no-synthetic", action="store_true")
    p = sub.add_parser("export-test"); task_args(p)
    p.add_argument("--annotations", required=True); p.add_argument("--frames")
    p.add_argument("--sessions", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("validate"); p.add_argument("dataset")
    p = sub.add_parser("synthesize", help="fal edits of train frames, re-labelled and filtered"); task_args(p); det_args(p)
    p.add_argument("--frames", required=True); p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True); p.add_argument("--max-images", type=int, default=20)
    p.add_argument("--exclude-sessions", nargs="*", default=[], help="Validation and test sessions (never edited)")
    p.add_argument("--conditions", nargs="*", help="Appearance edits. Default: TaskSpec.synthetic_targets")
    p.add_argument("--counterfactual", action="store_true",
                   help="Label-flipping edits from the TaskSpec rules (e.g. remove the vest)")
    p.add_argument("--verify", action="store_true", help="Claude vision check of every kept edit (ANTHROPIC_API_KEY)")
    p = sub.add_parser("run", help="ingest + annotate + export in one go"); task_args(p); ingest_args(p); det_args(p)
    p.add_argument("--out", required=True); p.add_argument("--held-out", nargs="*", default=[])
    p.add_argument("--val-sessions", nargs="*")

    a = ap.parse_args()
    if a.cmd == "ingest":
        fc = process_media(a.media, a.out, session_ids=_sessions(a.session), fps=a.fps,
                           max_frames=a.max_frames, min_sharpness=a.min_sharpness)
        print(json.dumps({"frames": len(fc.frames), "rejected": len(fc.rejected), "sessions": list(fc.sessions),
                          "frames_json": str(Path(a.out).resolve() / "frames.json")}, indent=2))
    elif a.cmd == "annotate":
        task = _load_task(a)
        fc = FrameCollection.model_validate_json(Path(a.frames).read_text())
        aset = auto_annotate(fc, task, detector=_detector(a), out_path=a.out)
        if a.preview:
            render_previews(aset, a.preview)
        print(json.dumps(aset.stats, indent=2))
    elif a.cmd == "correct":
        aset = AnnotationSet.model_validate_json(Path(a.annotations).read_text())
        fixes = [LabelCorrection.model_validate(c) for c in json.loads(Path(a.corrections).read_text())]
        out = apply_corrections(aset, fixes)
        Path(a.out or a.annotations).write_text(out.model_dump_json(indent=2))
        print(f"applied {len(fixes)} corrections")
    elif a.cmd in ("export", "export-test"):
        task = _load_task(a)
        aset = AnnotationSet.model_validate_json(Path(a.annotations).read_text())
        fc = FrameCollection.model_validate_json(Path(a.frames).read_text()) if a.frames else None
        if a.cmd == "export":
            m = export_dataset(aset, task, a.out, frames=fc, val_sessions=a.val_sessions,
                               held_out_sessions=a.held_out, include_synthetic=not a.no_synthetic)
        else:
            m = export_test_set(aset, task, a.out, a.sessions, frames=fc)
        _report(m)
    elif a.cmd == "synthesize":
        from .synthetic import generate_synthetic, merge
        task = _load_task(a)
        fc = FrameCollection.model_validate_json(Path(a.frames).read_text())
        aset = AnnotationSet.model_validate_json(Path(a.annotations).read_text())
        verifier = None
        if a.verify:
            from .synthetic import ClaudeEditVerifier
            verifier = ClaudeEditVerifier()
        syn = generate_synthetic(aset, fc.frames, task, a.out, detector=_detector(a), conditions=a.conditions,
                                 exclude_sessions=a.exclude_sessions, max_images=a.max_images,
                                 counterfactual=a.counterfactual, verifier=verifier)
        merged_frames = fc.model_copy(update={"frames": fc.frames + syn.frames})
        Path(a.out, "frames_merged.json").write_text(merged_frames.model_dump_json(indent=2))
        Path(a.out, "annotations_merged.json").write_text(merge(aset, syn).model_dump_json(indent=2))
        print(json.dumps({"kept": [{"frame_id": f.frame_id, "condition": f.generation["condition"],
                                    "verifier": f.generation.get("verifier")} for f in syn.frames],
                          "stats": syn.annotations.stats, "rejected": [r.model_dump() for r in syn.rejected],
                          "next": f"export --frames {a.out}/frames_merged.json --annotations {a.out}/annotations_merged.json"},
                         indent=2))
    elif a.cmd == "validate":
        r = validate_dataset(a.dataset)
        print(r.model_dump_json(indent=2))
        return 0 if r.ok else 1
    elif a.cmd == "run":
        task = _load_task(a)
        out = Path(a.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "task_spec.json").write_text(task.model_dump_json(indent=2))
        fc = process_media(a.media, out / "work", task=task, session_ids=_sessions(a.session), fps=a.fps,
                           max_frames=a.max_frames, min_sharpness=a.min_sharpness)
        aset = auto_annotate(fc, task, detector=_detector(a), out_path=out / "work" / "annotations.json")
        render_previews(aset, out / "work" / "preview")
        test_sessions = [s for s in a.held_out if s in fc.sessions]
        m = export_dataset(aset, task, out / "dataset", frames=fc, val_sessions=a.val_sessions,
                           held_out_sessions=a.held_out)
        if test_sessions:
            _report(export_test_set(aset, task, out / "test_set", test_sessions, frames=fc))
        _report(m)
        r = validate_dataset(m.data_yaml)
        print(f"validator: ok={r.ok} errors={r.errors[:5]}")
        return 0 if r.ok else 1
    return 0


def _report(m):
    print(json.dumps({
        "dataset_id": m.dataset_id, "variant": m.variant, "data_yaml": m.data_yaml,
        "split_strategy": m.split_strategy, "class_names": m.class_names,
        "splits": {k: v.model_dump() for k, v in m.splits.items()}, "warnings": m.warnings,
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())

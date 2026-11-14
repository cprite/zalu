"""VisionForge Data Engine: footage -> cleaned frames -> auto-labels -> YOLO dataset."""
from .annotate import YoloWorldDetector, apply_rules, auto_annotate
from .corrections import CorrectionError, apply_corrections
from .export import ExportError, export_dataset, export_test_set
from .ingest import process_media
from .schema import (Annotation, AnnotationSet, BoxLabel, DatasetManifest, Frame, FrameCollection,
                     LabelCorrection)
from .service import add_synthetic, get_annotations, process_test_upload, process_upload, save_corrections
from .validate import ValidationReport, check_leakage, validate_dataset

__all__ = [
    "process_upload", "process_test_upload", "add_synthetic", "save_corrections", "get_annotations",
    "process_media", "auto_annotate", "apply_rules", "YoloWorldDetector", "apply_corrections",
    "export_dataset", "export_test_set", "validate_dataset", "check_leakage",
    "Frame", "FrameCollection", "Annotation", "AnnotationSet", "BoxLabel", "LabelCorrection",
    "DatasetManifest", "ValidationReport", "ExportError", "CorrectionError",
]

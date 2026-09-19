"""Fine-tuning: the data checks that matter, and the memory math that decides
whether your card can do it."""

from punkai.train.data import (
    DatasetReport,
    Example,
    audit_dataset,
    canary_examples,
    check_memorization,
    contamination,
    near_duplicates,
    scan_pii,
)
from punkai.train.lora import LoraPlan, TrainingEstimate, estimate_training_gib, plan_summary

__all__ = [
    "DatasetReport",
    "Example",
    "LoraPlan",
    "TrainingEstimate",
    "audit_dataset",
    "canary_examples",
    "check_memorization",
    "contamination",
    "estimate_training_gib",
    "near_duplicates",
    "plan_summary",
    "scan_pii",
]

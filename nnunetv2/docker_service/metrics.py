from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import seg_metrics.seg_metrics as sg
from tqdm.contrib.concurrent import process_map


DEFAULT_METRICS = (
    "dice",
    "jaccard",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "vs",
    "hd",
    "hd95",
    "msd",
    "mdsd",
    "stdsd",
)


def _remove_nii_gz(filename: str) -> str:
    """Remove the .nii.gz extension from a filename."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]

    return Path(filename).stem


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalar values to native Python values."""
    if isinstance(value, np.generic):
        return value.item()

    return value


def _evaluate_case(
    task: tuple[str, str],
    labels: dict[str, int],
    metrics: tuple[str, ...],
    fully_connected: bool,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """
    Evaluate one prediction/GT pair.

    This function must remain at module level because process_map uses
    multiprocessing.
    """
    pred_path_string, gt_path_string = task

    pred_path = Path(pred_path_string)
    case_name = _remove_nii_gz(pred_path.name)

    result = sg.write_metrics(
        labels=list(labels.values()),
        gdth_path=gt_path_string,
        pred_path=pred_path_string,
        csv_file=None,
        metrics=list(metrics),
        verbose=False,
        fully_connected=fully_connected,
        TPTNFPFN=True,
    )

    # Some seg-metrics versions return a one-element list even when evaluating
    # a single prediction/GT pair.
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(
                f"Expected one result for {case_name}, got {len(result)}."
            )

        result = result[0]

    if not isinstance(result, dict):
        raise TypeError(
            f"Unexpected seg-metrics result for {case_name}: "
            f"{type(result).__name__}"
        )

    returned_labels = [
        int(_to_python_scalar(value))
        for value in result["label"]
    ]

    label_positions = {
        label_index: position
        for position, label_index in enumerate(returned_labels)
    }

    case_metrics: dict[str, dict[str, Any]] = {}

    for label_name, label_index in labels.items():
        if label_index not in label_positions:
            raise RuntimeError(
                f"Label {label_name!r} with index {label_index} was not "
                f"returned for case {case_name}."
            )

        position = label_positions[label_index]

        values: dict[str, Any] = {
            "index": label_index,
        }

        for metric_name, metric_values in result.items():
            if metric_name in {"label", "filename"}:
                continue

            if isinstance(metric_values, np.ndarray):
                metric_values = metric_values.tolist()

            if isinstance(metric_values, (list, tuple)):
                value = metric_values[position]
            else:
                value = metric_values

            values[metric_name] = _to_python_scalar(value)

        case_metrics[label_name] = values

    return case_name, case_metrics


def _as_finite_float(value: Any) -> float | None:
    """Convert a numeric value to float, excluding NaN and infinity."""
    if isinstance(value, (bool, np.bool_)):
        return None

    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def _calculate_mean_std(
    cases: Mapping[str, dict[str, dict[str, Any]]],
    labels: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Calculate per-label mean and population standard deviation across cases.

    NaN and infinite values are ignored independently for each metric.
    """
    mean_result: dict[str, Any] = {}
    std_result: dict[str, Any] = {}

    for label_name, label_index in labels.items():
        metric_names = {
            metric_name
            for case_metrics in cases.values()
            for metric_name in case_metrics[label_name]
            if metric_name != "index"
        }

        label_mean: dict[str, Any] = {
            "index": label_index,
        }

        label_std: dict[str, Any] = {
            "index": label_index,
        }

        for metric_name in sorted(metric_names):
            values = []

            for case_metrics in cases.values():
                value = _as_finite_float(
                    case_metrics[label_name].get(metric_name)
                )

                if value is not None:
                    values.append(value)

            if values:
                values_array = np.asarray(values, dtype=np.float64)

                label_mean[metric_name] = float(values_array.mean())
                label_std[metric_name] = float(values_array.std(ddof=0))
            else:
                label_mean[metric_name] = None
                label_std[metric_name] = None

        mean_result[label_name] = label_mean
        std_result[label_name] = label_std

    return mean_result, std_result


def _safe_division(
    numerator: int,
    denominator: int,
    empty_value: float,
) -> float:
    if denominator == 0:
        return empty_value

    return numerator / denominator


def _calculate_global_metrics(
    cases: Mapping[str, dict[str, dict[str, Any]]],
    labels: Mapping[str, int],
) -> dict[str, Any]:
    """
    Aggregate TP, FP and FN across all cases before calculating metrics.
    """
    global_result: dict[str, Any] = {}

    for label_name, label_index in labels.items():
        tp = sum(
            int(case_metrics[label_name]["TP"])
            for case_metrics in cases.values()
        )

        fp = sum(
            int(case_metrics[label_name]["FP"])
            for case_metrics in cases.values()
        )

        fn = sum(
            int(case_metrics[label_name]["FN"])
            for case_metrics in cases.values()
        )

        # If the label is completely absent from both GT and predictions,
        # treat the segmentation as perfect.
        empty_value = 1.0 if tp == 0 and fp == 0 and fn == 0 else 0.0

        global_result[label_name] = {
            "index": label_index,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "dice": _safe_division(
                numerator=2 * tp,
                denominator=2 * tp + fp + fn,
                empty_value=empty_value,
            ),
            "precision": _safe_division(
                numerator=tp,
                denominator=tp + fp,
                empty_value=empty_value,
            ),
            "recall": _safe_division(
                numerator=tp,
                denominator=tp + fn,
                empty_value=empty_value,
            ),
        }

    # Unweighted mean of the global per-label metrics.
    global_result["Mean"] = {
        metric_name: float(
            np.mean([
                global_result[label_name][metric_name]
                for label_name in labels
            ])
        )
        for metric_name in ("dice", "precision", "recall")
    }

    return global_result


def calculate_segmentation_metrics(
    pred_folder: str | Path,
    gt_folder: str | Path,
    labels: Mapping[str, int],
    num_workers: int,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    fully_connected: bool = True,
    chunksize: int = 1,
) -> dict[str, Any]:
    """
    Calculate segmentation metrics for all .nii.gz predictions.

    Prediction and GT files are matched by filename. For example:

        pred_folder/case_001.nii.gz
        gt_folder/case_001.nii.gz

    The returned case name is ``case_001``.

    Parameters
    ----------
    pred_folder:
        Folder containing prediction .nii.gz files.

    gt_folder:
        Folder containing matching ground-truth .nii.gz files.

    labels:
        Mapping from label names to voxel values, for example:

            {
                "organ": 1,
                "tumor": 2,
            }

    num_workers:
        Number of worker processes.

    metrics:
        Metrics requested from seg-metrics.

    fully_connected:
        Connectivity used for surface-distance metrics.

    chunksize:
        Number of cases assigned to each worker at a time.

    Returns
    -------
    dict
        Dictionary containing:

        - Cases: all per-case, per-label metrics
        - Mean: mean of each metric across cases
        - Std: population standard deviation across cases
        - Global: metrics calculated from globally aggregated TP/FP/FN
    """
    pred_folder = Path(pred_folder).expanduser().resolve()
    gt_folder = Path(gt_folder).expanduser().resolve()

    if not pred_folder.is_dir():
        raise NotADirectoryError(
            f"Prediction folder does not exist: {pred_folder}"
        )

    if not gt_folder.is_dir():
        raise NotADirectoryError(
            f"Ground-truth folder does not exist: {gt_folder}"
        )

    if not labels:
        raise ValueError("labels must not be empty.")

    normalized_labels = {
        str(label_name): int(label_index)
        for label_name, label_index in labels.items()
    }

    if len(set(normalized_labels.values())) != len(normalized_labels):
        raise ValueError("Each label must have a unique index.")

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1.")

    pred_files = sorted(pred_folder.glob("*.nii.gz"))

    if not pred_files:
        raise ValueError(
            f"No .nii.gz files were found in {pred_folder}"
        )

    tasks: list[tuple[str, str]] = []
    missing_gt_files: list[Path] = []

    for pred_path in pred_files:
        gt_path = gt_folder / pred_path.name

        if not gt_path.is_file():
            missing_gt_files.append(gt_path)
            continue

        tasks.append((str(pred_path), str(gt_path)))

    if missing_gt_files:
        missing = "\n".join(
            f"  {path}"
            for path in missing_gt_files
        )

        raise FileNotFoundError(
            "The following ground-truth files are missing:\n"
            f"{missing}"
        )

    worker = partial(
        _evaluate_case,
        labels=normalized_labels,
        metrics=tuple(metrics),
        fully_connected=fully_connected,
    )

    evaluated_cases = process_map(
        worker,
        tasks,
        max_workers=num_workers,
        chunksize=chunksize,
        desc="Calculating segmentation metrics",
        unit="case",
    )

    cases = dict(sorted(evaluated_cases))

    mean_metrics, std_metrics = _calculate_mean_std(
        cases=cases,
        labels=normalized_labels,
    )

    global_metrics = _calculate_global_metrics(
        cases=cases,
        labels=normalized_labels,
    )

    return {
        "Cases": cases,
        "Mean": mean_metrics,
        "Std": std_metrics,
        "Global": global_metrics,
    }
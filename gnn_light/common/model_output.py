import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score, roc_auc_score, precision_recall_curve,
)

SCHEMA_VERSION = "1.0"


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics(probs, targets, threshold: float = 0.5) -> Dict[str, float]:
    probs_np, targets_np = _to_numpy(probs), _to_numpy(targets)
    preds = (probs_np >= threshold).astype(int)  #pyright: ignore
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, preds, average="binary", zero_division=0,  #pyright: ignore
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": float(average_precision_score(targets_np, probs_np)),  #pyright: ignore
        "auroc": float(roc_auc_score(targets_np, probs_np)),  #pyright: ignore
        "n": int(len(targets_np)),  #pyright: ignore
        "n_positive": int(targets_np.sum()),  #pyright: ignore
    }


def best_fbeta_threshold(val_probs, val_targets, beta: float = 1.0, thresholds=None) -> float:
    val_probs_np, val_targets_np = _to_numpy(val_probs), _to_numpy(val_targets)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    b2 = beta ** 2
    best_score, best_thr = -1.0, 0.5
    for thr in thresholds:
        preds = (val_probs_np >= thr).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(
            val_targets_np, preds, average="binary", zero_division=0,  #pyright: ignore
        )
        denom = (b2 * precision) + recall
        fbeta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0
        if fbeta > best_score:
            best_score, best_thr = fbeta, float(thr)
    return best_thr


def best_f1_threshold(val_probs, val_targets, thresholds=None) -> float:
    return best_fbeta_threshold(val_probs, val_targets, beta=1.0, thresholds=thresholds)


def precision_at_k(probs, targets, ks=(100, 500, 1000, 5000)) -> Dict[str, Dict[str, Any]]:
    probs_np, targets_np = _to_numpy(probs), _to_numpy(targets)
    order = np.argsort(-probs_np)  #pyright: ignore
    out = {}
    for k in ks:
        k_eff = min(k, len(order))
        topk = order[:k_eff]
        caught = int(targets_np[topk].sum())  #pyright: ignore
        out[str(k)] = {"precision": float(caught / k_eff) if k_eff else 0.0, "caught": caught, "k": k_eff}
    return out


def save_predictions(path: str, *, val_pos, val_probs, val_targets, test_pos, test_probs, test_targets) -> str:
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "val_pos": torch.as_tensor(_to_numpy(val_pos), dtype=torch.long),
        "val_probs": torch.as_tensor(_to_numpy(val_probs), dtype=torch.float32),
        "val_targets": torch.as_tensor(_to_numpy(val_targets), dtype=torch.float32),
        "test_pos": torch.as_tensor(_to_numpy(test_pos), dtype=torch.long),
        "test_probs": torch.as_tensor(_to_numpy(test_probs), dtype=torch.float32),
        "test_targets": torch.as_tensor(_to_numpy(test_targets), dtype=torch.float32),
    }, path)
    return path


def export_top_flagged(path: str, *, test_pos, test_probs, test_targets, threshold: float, top_n: int = 5000) -> str:
    pos_np, probs_np, targets_np = _to_numpy(test_pos), _to_numpy(test_probs), _to_numpy(test_targets)
    order = np.argsort(-probs_np)[:top_n]  #pyright: ignore
    rows = []
    for rank, i in enumerate(order, start=1):
        score = float(probs_np[i])  #pyright: ignore
        actual = int(targets_np[i])  #pyright: ignore
        predicted = 1 if score >= threshold else 0
        rows.append({
            "rank": rank,
            "txn_pos": int(pos_np[i]),  #pyright: ignore
            "score": round(score, 4),
            "predicted_label": predicted,
            "actual_label": actual,
            "correct": predicted == actual,
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"threshold": threshold, "top_n": len(rows), "transactions": rows}, f)
    return path


def export_pr_curve(path: str, *, test_probs, test_targets, n_points: int = 150, best_threshold: float = None) -> str:  #pyright: ignore
    probs_np, targets_np = _to_numpy(test_probs), _to_numpy(test_targets)
    precision, recall, thresholds = precision_recall_curve(targets_np, probs_np)  #pyright: ignore
    precision, recall = precision[:-1], recall[:-1]
    n = len(thresholds)

    if n > n_points:
        idx = np.linspace(0, n - 1, n_points).astype(int)
    else:
        idx = np.arange(n)

    sorted_probs = np.sort(probs_np)  #pyright: ignore
    total = len(sorted_probs)

    def _point(thr, p, r, is_best=False):
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        alert_count = int(total - np.searchsorted(sorted_probs, thr, side="left"))
        return {
            "threshold": round(float(thr), 4), "precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(f1, 4), "alert_count": alert_count, "is_best": is_best,
        }

    points = [_point(thresholds[i], precision[i], recall[i]) for i in idx]

    if best_threshold is not None:
        m = compute_metrics(test_probs, test_targets, threshold=best_threshold)
        best_point = _point(best_threshold, m["precision"], m["recall"], is_best=True)
        points = [p for p in points if abs(p["threshold"] - best_point["threshold"]) > 1e-6]
        points.append(best_point)

    points.sort(key=lambda p: p["threshold"])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"n_test": total, "n_positive": int(targets_np.sum()), "points": points}, f) #pyright: ignore
    return path


@dataclass
class ModelRunResult:
    schema_version: str
    model_name: str
    dataset_size: str
    run_tag: str
    granularity: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    best_threshold: Optional[float] = None
    precision_at_k: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def to_xcom_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_size": self.dataset_size,
            "run_tag": self.run_tag,
            "status": self.status,
            "test_auprc": self.metrics.get("test", {}).get("auprc"),
            "test_auroc": self.metrics.get("test", {}).get("auroc"),
            "test_f1_at_best_threshold": self.metrics.get("test_best_threshold", {}).get("f1"),
            "best_threshold": self.best_threshold,
            "avg_ms_per_transaction": self.performance.get("avg_ms_per_transaction"),
            "predictions_path": self.artifacts.get("predictions_path"),
            "summary_json_path": self.artifacts.get("summary_json_path"),
        }


def finalize_run(
    *,
    model_name: str,
    dataset_size: str,
    run_tag: str,
    granularity: str,
    config: Dict[str, Any],
    val_probs, val_targets, val_pos,
    test_probs, test_targets, test_pos,
    started_at: datetime,
    output_dir: str = "./output",
    checkpoint_path: Optional[str] = None,
    log_path: Optional[str] = None,
    logger=None,
    test_inference_seconds: Optional[float] = None,
    top_flagged_n: int = 5000,
    threshold_beta: float = 2.0,
) -> ModelRunResult:
    finished_at = datetime.now(timezone.utc)
    predictions_path = os.path.join(output_dir, f"aml_{dataset_size}_{model_name}{run_tag}_predictions.pt")
    summary_path = os.path.join(output_dir, f"aml_{dataset_size}_{model_name}{run_tag}_run_summary.json")
    top_flagged_path = os.path.join(output_dir, f"aml_{dataset_size}_{model_name}{run_tag}_top_flagged.json")
    pr_curve_path = os.path.join(output_dir, f"aml_{dataset_size}_{model_name}{run_tag}_pr_curve.json")

    save_predictions(
        predictions_path,
        val_pos=val_pos, val_probs=val_probs, val_targets=val_targets,
        test_pos=test_pos, test_probs=test_probs, test_targets=test_targets,
    )

    thr = best_fbeta_threshold(val_probs, val_targets, beta=threshold_beta)
    metrics = {
        "val": compute_metrics(val_probs, val_targets, threshold=0.5),
        "test": compute_metrics(test_probs, test_targets, threshold=0.5),
        "test_best_threshold": compute_metrics(test_probs, test_targets, threshold=thr),
    }

    export_top_flagged(
        top_flagged_path, test_pos=test_pos, test_probs=test_probs, test_targets=test_targets,
        threshold=thr, top_n=top_flagged_n,
    )
    export_pr_curve(pr_curve_path, test_probs=test_probs, test_targets=test_targets, best_threshold=thr)

    n_test = len(_to_numpy(test_targets))  #pyright: ignore
    performance: Dict[str, Any] = {}
    if test_inference_seconds is not None:
        performance = {
            "test_set_size": n_test,
            "test_inference_seconds": test_inference_seconds,
            "avg_ms_per_transaction": (test_inference_seconds / n_test * 1000) if n_test else None,
            "throughput_transactions_per_sec": (n_test / test_inference_seconds) if test_inference_seconds > 0 else None,
        }

    artifacts = {
        "predictions_path": predictions_path, "summary_json_path": summary_path,
        "top_flagged_path": top_flagged_path, "pr_curve_path": pr_curve_path,
    }
    if checkpoint_path:
        artifacts["checkpoint_path"] = checkpoint_path
    if log_path:
        artifacts["log_path"] = log_path

    result = ModelRunResult(
        schema_version=SCHEMA_VERSION,
        model_name=model_name,
        dataset_size=dataset_size,
        run_tag=run_tag,
        granularity=granularity,
        status="success",
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=(finished_at - started_at).total_seconds(),
        config=config,
        metrics=metrics,
        best_threshold=thr,
        precision_at_k=precision_at_k(test_probs, test_targets),
        performance=performance,
        artifacts=artifacts,
    )
    result.save_json(summary_path)

    if logger is not None:
        logger.info(f"Saved predictions to {predictions_path}")
        logger.info(f"Saved run summary to {summary_path}")
        logger.info(f"Saved top-{top_flagged_n} flagged transactions to {top_flagged_path}")
        logger.info(f"Saved precision/recall-vs-threshold curve to {pr_curve_path}")
        if performance:
            logger.info(f"Test-set scoring: {performance['avg_ms_per_transaction']:.4f} ms/txn "
                        f"({performance['throughput_transactions_per_sec']:.1f} txn/s)")
        from gnn_light.common.logging_utils import log_json
        log_json(logger, "run_complete", result.to_xcom_dict())

    return result


def save_failed_run(
    *,
    model_name: str,
    dataset_size: str,
    run_tag: str,
    granularity: str,
    config: Dict[str, Any],
    started_at: datetime,
    error: Exception,
    output_dir: str = "./output",
    log_path: Optional[str] = None,
) -> ModelRunResult:
    finished_at = datetime.now(timezone.utc)
    summary_path = os.path.join(output_dir, f"aml_{dataset_size}_{model_name}{run_tag}_run_summary.json")
    artifacts = {"summary_json_path": summary_path}
    if log_path:
        artifacts["log_path"] = log_path
    result = ModelRunResult(
        schema_version=SCHEMA_VERSION,
        model_name=model_name,
        dataset_size=dataset_size,
        run_tag=run_tag,
        granularity=granularity,
        status="failed",
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=(finished_at - started_at).total_seconds(),
        config=config,
        artifacts=artifacts,
        error=f"{type(error).__name__}: {error}",
    )
    result.save_json(summary_path)
    return result

import argparse
import json
import os
from datetime import datetime, timezone
import torch

RECENT_EVENTS_MAX = 500

def _load_run(output_dir, size, model_name, run_tag=""):
    pred_path = os.path.join(output_dir, f"aml_{size}_{model_name}{run_tag}_predictions.pt")
    summary_path = os.path.join(output_dir, f"aml_{size}_{model_name}{run_tag}_run_summary.json")
    preds = torch.load(pred_path, weights_only=False)
    with open(summary_path) as f:
        summary = json.load(f)
    return preds, summary


def _load_summary(output_dir, size, model_name, run_tag=""):
    summary_path = os.path.join(output_dir, f"aml_{size}_{model_name}{run_tag}_run_summary.json")
    with open(summary_path) as f:
        return json.load(f)


def _cursor_path(output_dir):
    return os.path.join(output_dir, "stream_cursor.json")


def _events_path(output_dir):
    return os.path.join(output_dir, "stream_events.jsonl")


def _recent_events_path(output_dir):
    return os.path.join(output_dir, "stream_events_recent.jsonl")


def _live_model_path(output_dir, size, model_name, run_tag):
    ext = "json" if model_name == "xgboost" else "pt"
    return os.path.join(output_dir, f"aml_{size}_{model_name}{run_tag}_live_model.{ext}")


def _retrain_log_path(output_dir, size, model_name, run_tag):
    return os.path.join(output_dir, f"aml_{size}_{model_name}{run_tag}_retrain_log.json")


def _load_cursor(output_dir):
    path = _cursor_path(output_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "next_index": 0, "total_processed": 0, "total_alerts": 0,
        "true_positives": 0, "false_positives": 0, "true_negatives": 0, "false_negatives": 0,
        "since_last_retrain": 0, "pending_retrain_positions": [],
        "total_scoring_seconds": 0.0, "total_scored_online": 0,
    }


def _save_cursor(output_dir, cursor):
    os.makedirs(output_dir, exist_ok=True)
    with open(_cursor_path(output_dir), "w") as f:
        json.dump(cursor, f, indent=2)


def _append_recent_events(recent_path, new_events, cap=RECENT_EVENTS_MAX):
    existing = []
    if os.path.exists(recent_path):
        with open(recent_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(line)
    existing.extend(json.dumps(e) for e in new_events)
    existing = existing[-cap:]
    os.makedirs(os.path.dirname(recent_path) or ".", exist_ok=True)
    with open(recent_path, "w") as f:
        f.write("\n".join(existing) + ("\n" if existing else ""))


def _record_events(events_path, recent_events_path, cursor, *, positions, probs, targets, threshold,
                    model_name, model_version=1):
    start_seq = cursor["total_processed"]
    new_events = []
    with open(events_path, "a") as f:
        for i, pos in enumerate(positions):
            score = float(probs[i])
            actual = int(targets[i])
            decision = "FRAUD_ALERT" if score >= threshold else "PROCESSED"
            event = {
                "seq": start_seq + i + 1,
                "txn_pos": int(pos),
                "score": round(score, 4),
                "threshold": threshold,
                "decision": decision,
                "actual_label": actual,
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(event) + "\n")
            new_events.append(event)

            predicted_fraud = decision == "FRAUD_ALERT"
            if predicted_fraud and actual == 1:
                cursor["true_positives"] += 1
            elif predicted_fraud and actual == 0:
                cursor["false_positives"] += 1
            elif not predicted_fraud and actual == 1:
                cursor["false_negatives"] += 1
            else:
                cursor["true_negatives"] += 1
            if predicted_fraud:
                cursor["total_alerts"] += 1

    _append_recent_events(recent_events_path, new_events)


def _confusion_stats(cursor):
    tp, fp, fn = cursor["true_positives"], cursor["false_positives"], cursor["false_negatives"]
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    return tp, fp, fn, precision, recall


#REPLAY mode
def process_next_batch(
    *, output_dir="./output", size="Small", model_name="xgboost", run_tag="",
    batch_size=50, threshold=None, logger=None,
):
    preds, summary = _load_run(output_dir, size, model_name, run_tag)
    order = torch.argsort(preds["test_pos"]) 
    test_pos = preds["test_pos"][order].tolist()
    test_probs = preds["test_probs"][order].tolist()
    test_targets = preds["test_targets"][order].tolist()
    n = len(test_pos)

    thr = threshold if threshold is not None else summary.get("best_threshold", 0.5)

    cursor = _load_cursor(output_dir)
    start = cursor["next_index"]
    end = min(start + batch_size, n)

    events_path = _events_path(output_dir)
    recent_events_path = _recent_events_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    _record_events(
        events_path, recent_events_path, cursor, positions=test_pos[start:end], probs=test_probs[start:end],
        targets=test_targets[start:end], threshold=thr, model_name=model_name,
    )

    cursor["total_processed"] += (end - start)
    cursor["next_index"] = end
    cursor["stream_complete"] = end >= n
    cursor["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_cursor(output_dir, cursor)

    tp, fp, fn, precision, recall = _confusion_stats(cursor)
    status = {
        "model_name": model_name, "dataset_size": size, "run_tag": run_tag, "threshold": thr,
        "scoring_mode": "replay",
        "total_transactions": n,
        "total_processed": cursor["total_processed"],
        "total_alerts": cursor["total_alerts"],
        "alert_rate": (cursor["total_alerts"] / cursor["total_processed"]) if cursor["total_processed"] else 0.0,
        "running_precision": precision,
        "running_recall": recall,
        "true_positives": tp, "false_positives": fp,
        "true_negatives": cursor["true_negatives"], "false_negatives": fn,
        "stream_complete": cursor["stream_complete"],
        "last_updated": cursor["last_updated"],
    }
    with open(os.path.join(output_dir, "stream_status.json"), "w") as f:
        json.dump(status, f, indent=2)

    if logger:
        logger.info(
            f"[replay] Batch [{start}:{end}) of {n} scored | "
            f"+{end - start} txns, {cursor['total_alerts']} alerts total (rate {status['alert_rate']:.2%}) | "
            f"running precision={precision} recall={recall}"
        )
    return status


#ONLINE mode
_LIVE_SCORER_MODELS = {"xgboost", "gine_account_centric", "gine_transaction_centric"}


def _build_live_scorer(model_name, summary, *, csv_path=None, graph_path=None, live_model_path=None,
                        retrain_rounds=20, logger=None):
    checkpoint_path = summary["artifacts"]["checkpoint_path"]
    live_exists = bool(live_model_path and os.path.exists(live_model_path))
    resolved_ckpt = live_model_path if live_exists else checkpoint_path
    if logger:
        logger.info(
            f"Loading {'live/retrained' if live_exists else 'original pretrained'} weights from {resolved_ckpt} "
            f"(original checkpoint {checkpoint_path} is never modified by online scoring/retraining)."
        )

    if model_name == "xgboost":
        from simulation.live_scorer import XGBoostLiveScorer
        resolved_csv = csv_path or summary.get("config", {}).get("csv_path")
        if not resolved_csv:
            raise ValueError(
                "Online xgboost scoring needs the original transactions CSV"
            )
        return XGBoostLiveScorer(csv_path=resolved_csv, checkpoint_path=resolved_ckpt, retrain_rounds=retrain_rounds)

    resolved_graph = graph_path or summary.get("config", {}).get("graph_path")
    if not resolved_graph:
        raise ValueError(
            f"Online {model_name} scoring needs the graph .pt path"
        )
    if model_name == "gine_account_centric":
        from simulation.live_scorer import GINEAccountCentricLiveScorer
        return GINEAccountCentricLiveScorer(graph_path=resolved_graph, checkpoint_path=resolved_ckpt)
    if model_name == "gine_transaction_centric":
        from simulation.live_scorer import GINETransactionCentricLiveScorer
        return GINETransactionCentricLiveScorer(graph_path=resolved_graph, checkpoint_path=resolved_ckpt)

    raise ValueError(f"No live scorer implemented for model_name={model_name!r}")


def process_next_batch_online(
    *, output_dir="./output", size="Small", model_name="xgboost", run_tag="",
    batch_size=50, threshold=None, logger=None, csv_path=None, graph_path=None,
    retrain_every=1000, retrain_rounds=20, retrain_epochs=1,
):
    if model_name not in _LIVE_SCORER_MODELS:
        if logger:
            logger.warning(
                f"Online scoring isn't implemented for '{model_name}' Falling back to replay mode."
            )
        return process_next_batch(
            output_dir=output_dir, size=size, model_name=model_name, run_tag=run_tag,
            batch_size=batch_size, threshold=threshold, logger=logger,
        )

    summary = _load_summary(output_dir, size, model_name, run_tag)
    thr = threshold if threshold is not None else summary.get("best_threshold", 0.5)
    live_model_path = _live_model_path(output_dir, size, model_name, run_tag)
    retrain_log_path = _retrain_log_path(output_dir, size, model_name, run_tag)

    scorer = _build_live_scorer(
        model_name, summary, csv_path=csv_path, graph_path=graph_path,
        live_model_path=live_model_path, retrain_rounds=retrain_rounds, logger=logger,
    )
    scorer.load_retrain_log(retrain_log_path)

    cursor = _load_cursor(output_dir)
    n = scorer.n_test
    start = cursor["next_index"]
    end = min(start + batch_size, n)
    batch_positions = scorer.test_positions[start:end]

    probs, elapsed = scorer.score_positions(batch_positions)
    targets = scorer.get_targets(batch_positions)

    events_path = _events_path(output_dir)
    recent_events_path = _recent_events_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    _record_events(
        events_path, recent_events_path, cursor, positions=batch_positions, probs=probs, targets=targets,
        threshold=thr, model_name=model_name, model_version=scorer.version,
    )

    cursor["pending_retrain_positions"] = cursor.get("pending_retrain_positions", []) + [int(p) for p in batch_positions]
    cursor["since_last_retrain"] = cursor.get("since_last_retrain", 0) + len(batch_positions)
    if cursor["since_last_retrain"] >= retrain_every and cursor["pending_retrain_positions"]:
        if model_name == "xgboost":
            scorer.retrain_on(cursor["pending_retrain_positions"], logger=logger)
        else:
            scorer.retrain_on(cursor["pending_retrain_positions"], logger=logger, epochs=retrain_epochs)
        scorer.save_retrain_log(retrain_log_path)
        cursor["since_last_retrain"] = 0
        cursor["pending_retrain_positions"] = []

    scorer.save(live_model_path)

    cursor["total_processed"] += (end - start)
    cursor["next_index"] = end
    cursor["stream_complete"] = end >= n
    cursor["last_updated"] = datetime.now(timezone.utc).isoformat()
    cursor["total_scoring_seconds"] = cursor.get("total_scoring_seconds", 0.0) + elapsed
    cursor["total_scored_online"] = cursor.get("total_scored_online", 0) + len(batch_positions)
    _save_cursor(output_dir, cursor)

    tp, fp, fn, precision, recall = _confusion_stats(cursor)
    avg_scoring_ms = (
        cursor["total_scoring_seconds"] / cursor["total_scored_online"] * 1000
        if cursor["total_scored_online"] else None
    )
    last_retrain = scorer.retrain_log[-1] if scorer.retrain_log else None

    status = {
        "model_name": model_name, "dataset_size": size, "run_tag": run_tag, "threshold": thr,
        "scoring_mode": "online",
        "model_version": scorer.version,
        "total_transactions": n,
        "total_processed": cursor["total_processed"],
        "total_alerts": cursor["total_alerts"],
        "alert_rate": (cursor["total_alerts"] / cursor["total_processed"]) if cursor["total_processed"] else 0.0,
        "running_precision": precision,
        "running_recall": recall,
        "true_positives": tp, "false_positives": fp,
        "true_negatives": cursor["true_negatives"], "false_negatives": fn,
        "stream_complete": cursor["stream_complete"],
        "last_updated": cursor["last_updated"],
        "avg_scoring_ms_per_txn_live": avg_scoring_ms,
        "retrain_count": len(scorer.retrain_log),
        "retrain_every": retrain_every,
        "since_last_retrain": cursor["since_last_retrain"],
        "last_retrain": last_retrain,
    }
    with open(os.path.join(output_dir, "stream_status.json"), "w") as f:
        json.dump(status, f, indent=2)

    if logger:
        logger.info(
            f"[online v{scorer.version}] Batch [{start}:{end}) of {n} scored in {elapsed*1000:.2f}ms "
            f"({elapsed/max(len(batch_positions),1)*1000:.4f} ms/txn) | "
            f"{cursor['total_alerts']} alerts total (rate {status['alert_rate']:.2%}) | "
            f"running precision={precision} recall={recall}"
        )
    return status


def reset_stream(output_dir="./output", size=None, model_name=None, run_tag=""):
    for name in ("stream_cursor.json", "stream_events.jsonl", "stream_events_recent.jsonl", "stream_status.json"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
    if size and model_name:
        for path in (_live_model_path(output_dir, size, model_name, run_tag),
                     _retrain_log_path(output_dir, size, model_name, run_tag)):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--size", default="Small")
    parser.add_argument("--model-name", default="xgboost")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=None, help="Override the run's saved best_threshold.")
    parser.add_argument("--mode", choices=["replay", "online"], default="replay")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--retrain-every", type=int, default=1000)
    parser.add_argument("--retrain-rounds", type=int, default=20)
    parser.add_argument("--retrain-epochs", type=int, default=1)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    def _run_one():
        if args.mode == "online":
            return process_next_batch_online(
                output_dir=args.output_dir, size=args.size, model_name=args.model_name, run_tag=args.run_tag,
                batch_size=args.batch_size, threshold=args.threshold, csv_path=args.csv_path,
                graph_path=args.graph_path, retrain_every=args.retrain_every,
                retrain_rounds=args.retrain_rounds, retrain_epochs=args.retrain_epochs,
            )
        return process_next_batch(
            output_dir=args.output_dir, size=args.size, model_name=args.model_name,
            run_tag=args.run_tag, batch_size=args.batch_size, threshold=args.threshold,
        )

    if args.reset:
        reset_stream(args.output_dir, size=args.size, model_name=args.model_name, run_tag=args.run_tag)
        print("Stream state reset.")
    elif args.loop:
        while True:
            status = _run_one()
            extra = f" v{status['model_version']}" if status.get("scoring_mode") == "online" else ""
            print(f"[{status.get('scoring_mode','replay')}{extra}] "
                  f"processed={status['total_processed']}/{status['total_transactions']} "
                  f"alerts={status['total_alerts']} rate={status['alert_rate']:.2%}")
            if status["stream_complete"]:
                break
    else:
        status = _run_one()
        print(json.dumps(status, indent=2))

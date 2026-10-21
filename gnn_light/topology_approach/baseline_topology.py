import os
import sys
import time
from datetime import datetime, timezone

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gnn_light.common import get_logger
from gnn_light.common.model_output import finalize_run, save_failed_run

SIZE = "Medium"
RUN_TAG = ""
RUN_TAG_FILE = ""
GRAPH_PATH = f"./output/aml_{SIZE}_graph_homo_v2{RUN_TAG_FILE}.pt"
MODEL_NAME = "topology_baseline"



def reciprocity_flag(ref_edge_index, query_src, query_dst, num_nodes):
    key_mod = num_nodes + 1
    ref_src, ref_dst = ref_edge_index[0].numpy(), ref_edge_index[1].numpy()
    ref_keys = np.unique(ref_src.astype(np.int64) * key_mod + ref_dst.astype(np.int64))
    query_keys = query_dst.astype(np.int64) * key_mod + query_src.astype(np.int64)
    return np.isin(query_keys, ref_keys).astype(np.float32)


def percentile_rank(x):
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1)


def main(logger):
    started_at = datetime.now(timezone.utc)
    config = {"graph_path": GRAPH_PATH, "signals": [
        "endpoint_activity", "endpoint_burst", "amount_anomaly", "port_novelty", "reciprocity",
    ]}
    g = torch.load(GRAPH_PATH, weights_only=False)
    num_nodes = g["num_nodes"]
    node_x = g["node_x"].numpy()
    edge_index = g["edge_index"].numpy()
    edge_attr_numeric = g["edge_attr_numeric"].numpy()
    edge_label = g["edge_label"].numpy()
    split = g["split"].numpy()
    mp_edge_index = {0: g["mp_edge_index_train"], 1: g["mp_edge_index_val"], 2: g["mp_edge_index_test"]}

    log_paid = edge_attr_numeric[:, 0]
    amount_ratio = edge_attr_numeric[:, 3]
    in_port = edge_attr_numeric[:, 7]
    out_port = edge_attr_numeric[:, 8]

    src, dst = edge_index[0], edge_index[1]

    _t0 = time.perf_counter()

    # 0=out_count 1=in_count 2=out_sum 3=in_sum 4=out_mean 5=in_mean
    # 6=out_unique_cp 7=in_unique_cp 8=out_max_ratio 9=in_max_ratio
    endpoint_activity = node_x[src, 0] + node_x[dst, 1] + node_x[src, 6] + node_x[dst, 7]
    endpoint_burst = node_x[src, 8] + node_x[dst, 9]
    amount_anomaly = log_paid + np.abs(amount_ratio - 1.0)
    port_novelty = in_port + out_port

    reciprocity = np.zeros(len(src), dtype=np.float32)
    for s in [0, 1, 2]:
        mask = split == s
        reciprocity[mask] = reciprocity_flag(mp_edge_index[s], src[mask], dst[mask], num_nodes)

    val_mask, test_mask = split == 1, split == 2
    val_y, test_y = edge_label[val_mask], edge_label[test_mask]

    named_signals = {
        "endpoint_activity": endpoint_activity,
        "endpoint_burst": endpoint_burst,
        "amount_anomaly": amount_anomaly,
        "port_novelty": port_novelty,
        "reciprocity": reciprocity,
    }

    # diagnose each signal. A signal with val AUROC < 0.5 gets flipped
    logger.info("Per-signal diagnostics (val set, RAW direction before flipping):")
    oriented_signals = {}
    for name, sig in named_signals.items():
        raw_auc = roc_auc_score(val_y, sig[val_mask])
        flip = raw_auc < 0.5
        oriented = -sig if flip else sig
        oriented_auc = roc_auc_score(val_y, oriented[val_mask])
        oriented_signals[name] = oriented
        logger.info(f"  {name:20s} raw AUROC {raw_auc:.3f}"
              f"{' -> flipped, now ' + format(oriented_auc, '.3f') if flip else ' (kept as-is)'}")
    config["signal_polarity"] = {name: ("flipped" if roc_auc_score(val_y, s[val_mask]) < 0.5 else "kept")
                                  for name, s in named_signals.items()}

    ranks = np.stack([percentile_rank(s) for s in oriented_signals.values()], axis=1)
    topology_score = ranks.mean(axis=1)
    scoring_seconds = time.perf_counter() - _t0

    # evaluation
    val_score, test_score = topology_score[val_mask], topology_score[test_mask]
    val_pos, test_pos = np.where(val_mask)[0], np.where(test_mask)[0]

    result = finalize_run(
        model_name=MODEL_NAME, dataset_size=SIZE, run_tag=RUN_TAG, granularity="edge", config=config,
        val_probs=val_score, val_targets=val_y, val_pos=val_pos,
        test_probs=test_score, test_targets=test_y, test_pos=test_pos,
        started_at=started_at, checkpoint_path=None, log_path=logger.log_path, logger=logger,
        test_inference_seconds=scoring_seconds * (len(test_score) / len(topology_score)),
    )
    logger.info(f"VAL  | {result.metrics['val']}")
    logger.info(f"TEST | {result.metrics['test']}")
    logger.info(f"Best-F1 threshold (from val): {result.best_threshold:.4f}")
    logger.info(f"TEST (best-F1 threshold) | {result.metrics['test_best_threshold']}")
    for k, stats in result.precision_at_k.items():
        logger.info(f"Precision among top-{k:>5} ranked test transactions: {stats['precision']:.3f} "
                    f"({stats['caught']} real laundering txns caught)")
    return result


if __name__ == "__main__":
    _logger = get_logger(MODEL_NAME, SIZE, RUN_TAG)
    _started_at = datetime.now(timezone.utc)
    try:
        main(_logger)
    except Exception as exc:
        _logger.exception("Run failed")
        save_failed_run(
            model_name=MODEL_NAME, dataset_size=SIZE, run_tag=RUN_TAG, granularity="edge",
            config={"graph_path": GRAPH_PATH}, started_at=_started_at, error=exc, log_path=_logger.log_path  # pyright: ignore
        )
        raise

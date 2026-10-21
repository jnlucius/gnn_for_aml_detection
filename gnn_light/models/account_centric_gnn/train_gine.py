import os
import sys
import gc
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from gnn_light.common import get_logger
from gnn_light.common.model_output import finalize_run, save_failed_run
from gnn_light.models.account_centric_gnn import model as gmodel

SIZE = "Medium"
RUN_TAG = "_"
RUN_TAG_FILE = ""
GRAPH_PATH = f"./output/aml_{SIZE}_graph_homo_v2{RUN_TAG_FILE}.pt"
CKPT_PATH = f"./output/aml_{SIZE}_ckpt_gine{RUN_TAG_FILE}.pt"
HIDDEN = gmodel.HIDDEN
EPOCHS = 30
BATCH_SIZE = gmodel.BATCH_SIZE
NUM_NEIGHBORS = gmodel.NUM_NEIGHBORS
NUM_WORKERS = gmodel.NUM_WORKERS
TARGET_POS_RATIO = gmodel.TARGET_POS_RATIO
CAT_EMB_DIM = gmodel.CAT_EMB_DIM
NUM_GNN_LAYERS = gmodel.NUM_GNN_LAYERS
EDGE_UPDATES = gmodel.EDGE_UPDATES
EVAL_ONLY = False
MODEL_NAME = "gine_account_centric"


def main(logger):
    started_at = datetime.now(timezone.utc)
    config = {
        "graph_path": GRAPH_PATH, "hidden": HIDDEN, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "num_neighbors": NUM_NEIGHBORS, "target_pos_ratio": TARGET_POS_RATIO, "cat_emb_dim": CAT_EMB_DIM,
        "num_gnn_layers": NUM_GNN_LAYERS, "edge_updates": EDGE_UPDATES, "eval_only": EVAL_ONLY,
    }
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    g = gmodel.load_graph(GRAPH_PATH)
    logger.info(f"{g['num_nodes']:,} nodes, {g['edge_index'].size(1):,} total edges, "
                f"{g['edge_attr_numeric'].size(1)} numeric edge features")

    views = {}
    loaders = {}
    for name in ["train", "val", "test"]:
        views[name] = gmodel.build_split_view(g, name, target_pos_ratio=TARGET_POS_RATIO)
        if views[name]["oversample_info"]:
            effective_ratio, num_pos, num_neg, repeat_factor = views[name]["oversample_info"]
            logger.info(f"Train oversampling: {num_pos:,} pos / {num_neg:,} neg (raw) -> "
                  f"{effective_ratio:.2%} positive per batch ({repeat_factor}x repeat)")
        loaders[name] = gmodel.make_loader(
            views[name], batch_size=BATCH_SIZE, shuffle=(name == "train"), num_workers=NUM_WORKERS,
        )
    pos_lookup = {name: views[name]["pos_lookup"] for name in views}
    N_NUMERIC = views["train"]["N_NUMERIC"]
    cat_card = views["train"]["cat_card"]
    pos_rate = g["edge_label"][g["split"] == 0].mean().item()
    del g
    gc.collect()

    encoder = gmodel.GINeEdgeEncoder(
        views["train"]["data"].x.size(1), N_NUMERIC, cat_card, HIDDEN, num_layers=NUM_GNN_LAYERS,
    ).to(device)
    classifier = gmodel.EdgeClassifier(HIDDEN, N_NUMERIC, cat_card).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=0.003, weight_decay=1e-5,
    )

    pos_weight = torch.tensor([1.0], device=device)
    logger.info(f"train positive rate: {pos_rate:.4%} (raw), oversampled toward {TARGET_POS_RATIO:.0%} per batch, "
          f"pos_weight={pos_weight.item():.1f}")

    def run_epoch(loader, split_pos_lookup, train: bool):
        encoder.train(train)
        classifier.train(train)
        all_probs, all_targets, all_pos = [], [], []
        total_loss, total_n = 0.0, 0

        for batch in loader:
            batch_input_id = batch.input_id
            batch = batch.to(device)
            packed = batch.edge_label
            target = packed[:, 0]
            numeric_feats = packed[:, 1:1 + N_NUMERIC]
            cat_feats = packed[:, 1 + N_NUMERIC:].long()
            global_pos = split_pos_lookup[batch_input_id.cpu()]

            if train:
                optimizer.zero_grad()

            h = encoder(batch.x, batch.edge_index, batch.edge_attr)
            src, dst = batch.edge_label_index
            logits = classifier(h[src], h[dst], numeric_feats, cat_feats)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * target.size(0)
            total_n += target.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(target.detach().cpu())
            all_pos.append(global_pos)

        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)
        all_pos = torch.cat(all_pos)
        preds = (all_probs > 0.5).float()
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, preds, average="binary", zero_division=0,
        )
        ap = average_precision_score(all_targets, all_probs)
        auc = roc_auc_score(all_targets, all_probs)
        return total_loss / total_n, precision, recall, f1, ap, auc, all_probs, all_targets, all_pos

    start_epoch = 1
    best_ap, waited, PATIENCE = -1.0, 0, 5
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        classifier.load_state_dict(ckpt["classifier"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_ap = ckpt.get("best_ap", -1.0)
        logger.info(f"Resumed from checkpoint at epoch {ckpt['epoch']} (best val AUPRC so far: {best_ap:.4f})")
    elif EVAL_ONLY:
        raise FileNotFoundError(f"EVAL_ONLY=True but no checkpoint found at {CKPT_PATH}")

    if EVAL_ONLY:
        logger.info("EVAL_ONLY=True. Evaluating loaded checkpoint directly.")
    else:
        for epoch in range(start_epoch, start_epoch + EPOCHS):
            train_loss, p, r, f1, ap, auc, _, _, _ = run_epoch(loaders["train"], pos_lookup["train"], train=True)
            logger.info(f"Epoch {epoch} | train loss {train_loss:.4f} | "
                  f"P {p:.3f} R {r:.3f} F1 {f1:.3f} | AUPRC {ap:.3f} AUROC {auc:.3f}")

            val_loss, p, r, f1, val_ap, auc, _, _, _ = run_epoch(loaders["val"], pos_lookup["val"], train=False)
            logger.info(f"          | val   loss {val_loss:.4f} | "
                  f"P {p:.3f} R {r:.3f} F1 {f1:.3f} | AUPRC {val_ap:.3f} AUROC {auc:.3f}")

            if val_ap > best_ap:
                best_ap, waited = val_ap, 0
                torch.save({
                    "encoder": encoder.state_dict(), "classifier": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(), "epoch": epoch, "best_ap": best_ap,
                }, CKPT_PATH)
                logger.info(f"  -> new best val AUPRC, checkpoint saved")
            else:
                waited += 1
                if waited >= PATIENCE:
                    logger.info(f"Early stopping (no val AUPRC improvement in {PATIENCE} epochs).")
                    break

    ckpt = torch.load(CKPT_PATH, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    classifier.load_state_dict(ckpt["classifier"])

    _, _, _, _, _, _, val_probs, val_targets, val_pos = run_epoch(loaders["val"], pos_lookup["val"], train=False)
    _t0 = time.perf_counter()
    test_loss, p, r, f1, ap, auc, test_probs, test_targets, test_pos = run_epoch(
        loaders["test"], pos_lookup["test"], train=False,
    )
    test_inference_seconds = time.perf_counter() - _t0
    logger.info(f"\nTEST (best val-AUPRC checkpoint, epoch {ckpt['epoch']}, threshold=0.5) | loss {test_loss:.4f} | "
          f"P {p:.3f} R {r:.3f} F1 {f1:.3f} | AUPRC {ap:.3f} AUROC {auc:.3f}")

    result = finalize_run(
        model_name=MODEL_NAME, dataset_size=SIZE, run_tag=RUN_TAG, granularity="edge", config=config,
        val_probs=val_probs, val_targets=val_targets, val_pos=val_pos,
        test_probs=test_probs, test_targets=test_targets, test_pos=test_pos,
        started_at=started_at, checkpoint_path=CKPT_PATH, log_path=logger.log_path, logger=logger,
        test_inference_seconds=test_inference_seconds,
    )
    logger.info(f"TEST (threshold=0.5) | {result.metrics['test']}")
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
            config={"graph_path": GRAPH_PATH}, started_at=_started_at, error=exc, log_path=_logger.log_path  # pyright: ignore[reportAttributeAccessIssue]
        ) 
        raise

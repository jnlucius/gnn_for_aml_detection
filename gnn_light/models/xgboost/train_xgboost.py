import os
import sys
import time
from datetime import datetime, timezone

import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from gnn_light.common import get_logger
from gnn_light.common.model_output import finalize_run, save_failed_run
from gnn_light.models.xgboost import features

SIZE = "Medium"
RUN_TAG = ""
RUN_TAG_FILE = ""
CSV_PATH = f"./datasets/ibm_aml/HI-{SIZE}_Trans.csv"
CKPT_PATH = f"./output/aml_{SIZE}_xgboost_model{RUN_TAG_FILE}.json"
EVAL_ONLY = False
MODEL_NAME = "xgboost"


def main(logger):
    started_at = datetime.now(timezone.utc)
    config = {
        "csv_path": CSV_PATH, "eval_only": EVAL_ONLY, "n_estimators": 500, "max_depth": 6,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "early_stopping_rounds": 60,
    }

    logger.info("Loading CSV and building features...")
    data = features.build_all(CSV_PATH)
    feat_df, label, split, n_edges = data["feat_df"], data["label"], data["split"], data["n_edges"]
    logger.info(f"{n_edges:,} transactions, {label.mean():.4%} laundering rate, "
                f"{data['num_nodes']:,} unique accounts")
    logger.info(f"Split sizes: train={(split == 0).sum()} val={(split == 1).sum()} test={(split == 2).sum()}")
    for name, s in zip(["train", "val", "test"], [0, 1, 2]):
        logger.info(f"  {name} laundering rate: {label[split == s].mean():.4%}")
    logger.info(f"Feature table: {feat_df.shape[1]} columns, {feat_df.shape[0]:,} rows")

    train_mask = split == 0
    X_train, y_train = feat_df[train_mask], label[train_mask]
    X_val, y_val = feat_df[split == 1], label[split == 1]
    X_test, y_test = feat_df[split == 2], label[split == 2]

    # Train XGBoost
    pos_rate = y_train.mean()
    scale_pos_weight = ((1 - pos_rate) / pos_rate) ** 0.5
    logger.info(f"train positive rate: {pos_rate:.4%}, scale_pos_weight: {scale_pos_weight:.1f}")
    config["scale_pos_weight"] = float(scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=60,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    if EVAL_ONLY:
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(f"EVAL_ONLY=True but no checkpoint found at {CKPT_PATH}")
        logger.info(f"EVAL_ONLY=True. Evaluating from saved checkpoint.")
        warmup_model = xgb.XGBClassifier(
            n_estimators=1, enable_categorical=True, tree_method="hist", random_state=42,
        )
        warmup_model.fit(X_train.iloc[:2], y_train[:2])
        warmup_model.load_model(CKPT_PATH)
        model = warmup_model
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        model.save_model(CKPT_PATH)
        logger.info(f"Saved model checkpoint to {CKPT_PATH}")


    val_probs = model.predict_proba(X_val)[:, 1]
    _t0 = time.perf_counter()
    test_probs = model.predict_proba(X_test)[:, 1]
    test_inference_seconds = time.perf_counter() - _t0

    logger.info("Top 15 features by gain:")
    importances = model.get_booster().get_score(importance_type="gain")
    for feat, score in sorted(importances.items(), key=lambda x: -x[1])[:15]:  # pyright: ignore
        logger.info(f"  {feat}: {score:.1f}")

    result = finalize_run(
        model_name=MODEL_NAME, dataset_size=SIZE, run_tag=RUN_TAG, granularity="edge", config=config,
        val_probs=val_probs, val_targets=y_val, val_pos=X_val.index.to_numpy(),
        test_probs=test_probs, test_targets=y_test, test_pos=X_test.index.to_numpy(),
        started_at=started_at, checkpoint_path=CKPT_PATH, log_path=logger.log_path, logger=logger,
        test_inference_seconds=test_inference_seconds,
    )

    logger.info(
        f"VAL  (threshold=0.5) | {result.metrics['val']}"
    )
    logger.info(
        f"TEST (threshold=0.5) | {result.metrics['test']}"
    )
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
            config={"csv_path": CSV_PATH}, started_at=_started_at, error=exc, log_path=_logger.log_path  # pyright: ignore[reportAttributeAccessIssue]
        )
        raise

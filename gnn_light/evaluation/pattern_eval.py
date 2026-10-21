import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score

SIZE = "Medium"
PATTERN_PATH = f"./output/aml_{SIZE}_pattern_labels.pt"
MODEL_PREDICTION_PATHS = {
    # "GINE": f"./output/aml_{SIZE}_gine_predictions.pt",
    # "GINE_shuffled": f"./output/aml_Small_gine_account_centric_BETA2_SHUFFLED_predictions.pt",
    # "XGBoost": f"./output/aml_{SIZE}_xgboost_predictions.pt",
    # "GINE_txnode": f"./output/aml_{SIZE}_txnode_gine_predictions.pt",
    # "GINE_txnode_NOAMOUNT": f"./output/aml_{SIZE}_gine_transaction_centric_BETA2_NOAMOUNT_predictions.pt"
}


def load_predictions(path):
    try:
        return torch.load(path, weights_only=False)
    except FileNotFoundError:
        return None


def main():
    patterns = torch.load(PATTERN_PATH, weights_only=False)
    pattern_of_edge = patterns["pattern_of_edge"]
    n_edges = patterns["n_edges"]
    pattern_arr = np.array([p if p is not None else "NONE" for p in pattern_of_edge])

    for model_name, path in MODEL_PREDICTION_PATHS.items():
        preds = load_predictions(path)
        if preds is None:
            print(f"\n[skipping {model_name}: no predictions file found at {path}]")
            continue

        print("\n" + "=" * 70)
        print(f"{model_name}")
        print("=" * 70)

        for split_name in ["val", "test"]:
            pos = preds[f"{split_name}_pos"].numpy()
            probs = preds[f"{split_name}_probs"].numpy()
            targets = preds[f"{split_name}_targets"].numpy()
            assert pos.max() < n_edges, "position out of range"

            split_patterns = pattern_arr[pos]

            auprc = average_precision_score(targets, probs)
            auroc = roc_auc_score(targets, probs)
            print(f"\n--- {split_name.upper()} --- overall AUPRC {auprc:.4f}  AUROC {auroc:.4f}")

            thresholds = np.linspace(0.01, 0.99, 99)
            best_f1, best_thr = -1.0, 0.5
            for thr in thresholds:
                pred_bin = (probs > thr).astype(np.float32)
                _, _, f1, _ = precision_recall_fscore_support(targets, pred_bin, average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            pred_bin = (probs > best_thr).astype(np.float32)

            print(f"Per-pattern RECALL at threshold={best_thr:.2f} (flagged = predicted positive):")
            for p in sorted(np.unique(split_patterns)):
                pmask = split_patterns == p
                n = pmask.sum()
                rate = pred_bin[pmask].mean()
                label = "flag rate (false-positive check)" if p == "NONE" else "RECALL"
                print(f"  {p:15s} n={n:6d}  {label}: {rate:.2%}")

            print(f"\nPer-pattern presence in TOP-1000 ranked transactions (rank-based, threshold-free):")
            top_idx = np.argsort(-probs)[:1000]
            top_patterns = split_patterns[top_idx]
            for p in sorted(np.unique(top_patterns)):
                if p == "NONE":
                    continue
                n_true = (split_patterns == p).sum()
                n_in_top = (top_patterns == p).sum()
                print(f"  {p:15s} {n_in_top:4d} of top-1000 are this pattern "
                      f"(out of {n_true} true instances in {split_name})")


if __name__ == "__main__":
    main()

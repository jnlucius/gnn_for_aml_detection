import numpy as np
import torch

SIZE = "Small"
GRAPH_PATH = f"./output/aml_{SIZE}_graph_homo_v2.pt"
OUT_PATH = f"./output/aml_{SIZE}_graph_homo_v2_SHUFFLED.pt"
SEED = 0


def degree_preserving_shuffle(src, dst, rng):
    perm = rng.permutation(len(dst))
    shuffled_dst = dst[perm].copy()

    self_loop_mask = src == shuffled_dst
    n_loops = int(self_loop_mask.sum())
    if n_loops > 0:
        idx_loops = np.where(self_loop_mask)[0]
        candidates = np.where(~self_loop_mask)[0]
        idx_other = rng.choice(candidates, size=n_loops, replace=False)
        shuffled_dst[idx_loops], shuffled_dst[idx_other] = (
            shuffled_dst[idx_other].copy(), shuffled_dst[idx_loops].copy(),
        )
    return shuffled_dst


def verify_degree_preserved(orig_src, orig_dst, new_src, new_dst, num_nodes, label):
    orig_out = np.bincount(orig_src, minlength=num_nodes)
    orig_in = np.bincount(orig_dst, minlength=num_nodes)
    new_out = np.bincount(new_src, minlength=num_nodes)
    new_in = np.bincount(new_dst, minlength=num_nodes)
    out_ok = np.array_equal(orig_out, new_out)
    in_ok = np.array_equal(orig_in, new_in)
    identity_changed = not np.array_equal(orig_dst, new_dst)
    print(f"  [{label}] out-degree preserved: {out_ok} | in-degree preserved: {in_ok} | "
          f"edge identities actually changed: {identity_changed}")
    assert out_ok and in_ok, f"{label}: degree sequence was NOT preserved"
    return out_ok and in_ok


def main():
    rng = np.random.default_rng(SEED)
    g = torch.load(GRAPH_PATH, weights_only=False)
    split = g["split"].numpy()
    num_nodes = g["num_nodes"]

    mp_train = g["mp_edge_index_train"].numpy()
    mp_test = g["mp_edge_index_test"].numpy()
    assert torch.equal(g["mp_edge_index_train"], g["mp_edge_index_val"]), (
        "assumes mp_edge_index_train == mp_edge_index_val, as in the "
    )
    n_train_edges = mp_train.shape[1]

    assert mp_test.shape[1] >= n_train_edges
    val_period_src = mp_test[0, n_train_edges:]
    val_period_dst = mp_test[1, n_train_edges:]

    print("Shuffling train-period message-passing edges...")
    shuffled_train_dst = degree_preserving_shuffle(mp_train[0], mp_train[1], rng)
    verify_degree_preserved(mp_train[0], mp_train[1], mp_train[0], shuffled_train_dst, num_nodes, "train-period")

    print("Shuffling val-period message-passing edges...")
    shuffled_val_period_dst = degree_preserving_shuffle(val_period_src, val_period_dst, rng)
    verify_degree_preserved(
        val_period_src, val_period_dst, val_period_src, shuffled_val_period_dst, num_nodes, "val-period",
    )

    shuffled_mp_train = torch.tensor(np.stack([mp_train[0], shuffled_train_dst]), dtype=torch.long)
    shuffled_mp_val = shuffled_mp_train.clone()
    shuffled_mp_test = torch.tensor(np.stack([
        np.concatenate([mp_train[0], val_period_src]),
        np.concatenate([shuffled_train_dst, shuffled_val_period_dst]),
    ]), dtype=torch.long)

    print(f"\nShuffled shapes - train: {shuffled_mp_train.shape}, val: {shuffled_mp_val.shape}, "
          f"test: {shuffled_mp_test.shape} (should match originals: "
          f"{g['mp_edge_index_train'].shape}, {g['mp_edge_index_val'].shape}, {g['mp_edge_index_test'].shape})")


    out = dict(g)
    out["mp_edge_index_train"] = shuffled_mp_train
    out["mp_edge_index_val"] = shuffled_mp_val
    out["mp_edge_index_test"] = shuffled_mp_test
    torch.save(out, OUT_PATH)
    print(f"\nSaved degree-preserving shuffled graph to {OUT_PATH}")

if __name__ == "__main__":
    main()

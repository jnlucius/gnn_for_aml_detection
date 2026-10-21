import gc
import time
import numpy as np
import pandas as pd
import torch
from numba import njit

SIZE = "Medium"
CSV_PATH = f"./datasets/ibm_aml/HI-{SIZE}_Trans.csv"
OUT_PATH = f"./output/aml_{SIZE}_graph_txnode_v1.pt"

TIME_WINDOW_SECONDS = 4 * 86400
MAX_FANOUT = 10

DTYPES = {
    "From Bank": "category", "Account": "category", "To Bank": "category", "Account.1": "category",
    "Amount Received": "float32", "Receiving Currency": "category",
    "Amount Paid": "float32", "Payment Currency": "category", "Payment Format": "category",
    "Is Laundering": "int8",
}


@njit
def count_flow_through_edges(
    dep_key_sorted, dep_t_sorted,
    wd_key_sorted, wd_t_sorted,
    wd_group_keys, wd_boundaries,
    time_window, max_fanout,
):
    total = 0
    n_groups = len(wd_group_keys)
    for i in range(len(dep_key_sorted)):
        acct = dep_key_sorted[i]
        lo_g, hi_g = 0, n_groups
        while lo_g < hi_g:
            mid = (lo_g + hi_g) // 2
            if wd_group_keys[mid] < acct:
                lo_g = mid + 1
            else:
                hi_g = mid
        if lo_g >= n_groups or wd_group_keys[lo_g] != acct:
            continue
        lo, hi = wd_boundaries[lo_g], wd_boundaries[lo_g + 1]
        t_dep = dep_t_sorted[i]
        a, b = lo, hi
        while a < b:
            mid = (a + b) // 2
            if wd_t_sorted[mid] <= t_dep:
                a = mid + 1
            else:
                b = mid
        start = a
        t_end = t_dep + time_window
        a, b = start, hi
        while a < b:
            mid = (a + b) // 2
            if wd_t_sorted[mid] <= t_end:
                a = mid + 1
            else:
                b = mid
        end = a
        total += min(end, start + max_fanout) - start
    return total


@njit
def match_flow_through_edges(
    dep_key_sorted, dep_t_sorted, dep_id_sorted,
    wd_key_sorted, wd_t_sorted, wd_id_sorted,
    wd_group_keys, wd_boundaries,
    time_window, max_fanout, max_edges,
):
    edge_src = np.empty(max_edges, dtype=np.int64)
    edge_dst = np.empty(max_edges, dtype=np.int64)
    edge_gap = np.empty(max_edges, dtype=np.int64)
    n_edges = 0
    n_groups = len(wd_group_keys)

    for i in range(len(dep_key_sorted)):
        acct = dep_key_sorted[i]

        lo_g, hi_g = 0, n_groups
        while lo_g < hi_g:
            mid = (lo_g + hi_g) // 2
            if wd_group_keys[mid] < acct:
                lo_g = mid + 1
            else:
                hi_g = mid
        if lo_g >= n_groups or wd_group_keys[lo_g] != acct:
            continue
        lo, hi = wd_boundaries[lo_g], wd_boundaries[lo_g + 1]

        t_dep = dep_t_sorted[i]
        a, b = lo, hi
        while a < b:
            mid = (a + b) // 2
            if wd_t_sorted[mid] <= t_dep:
                a = mid + 1
            else:
                b = mid
        start = a

        t_end = t_dep + time_window
        a, b = start, hi
        while a < b:
            mid = (a + b) // 2
            if wd_t_sorted[mid] <= t_end:
                a = mid + 1
            else:
                b = mid
        end = a

        cand_end = min(end, start + max_fanout)
        for j in range(start, cand_end):
            if n_edges >= max_edges:
                return edge_src[:n_edges], edge_dst[:n_edges], edge_gap[:n_edges]
            edge_src[n_edges] = dep_id_sorted[i]
            edge_dst[n_edges] = wd_id_sorted[j]
            edge_gap[n_edges] = wd_t_sorted[j] - t_dep
            n_edges += 1

    return edge_src[:n_edges], edge_dst[:n_edges], edge_gap[:n_edges]


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, dtype=DTYPES, parse_dates=["Timestamp"])  # pyright: ignore[reportArgumentType]
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    print(f"{n:,} transactions, {df['Is Laundering'].mean():.4%} laundering rate")

    # Global account index
    acct_codes, acct_cats = pd.factorize(pd.concat([df["Account"], df["Account.1"]], ignore_index=True))
    acct_src, acct_dst = acct_codes[:n], acct_codes[n:]
    bank_codes, bank_cats = pd.factorize(pd.concat([df["From Bank"], df["To Bank"]], ignore_index=True))
    bank_src, bank_dst = bank_codes[:n], bank_codes[n:]
    num_accounts_local = len(acct_cats)
    withdrawal_key = bank_src.astype(np.int64) * num_accounts_local + acct_src.astype(np.int64)
    deposit_key = bank_dst.astype(np.int64) * num_accounts_local + acct_dst.astype(np.int64)
    print(f"{len(np.union1d(withdrawal_key, deposit_key)):,} unique accounts")

    ts = df["Timestamp"].astype("datetime64[s]").astype("int64").to_numpy()
    amount_paid = df["Amount Paid"].to_numpy(dtype=np.float32)
    amount_received = df["Amount Received"].to_numpy(dtype=np.float32)

    # Node features
    all_currencies = pd.Index(df["Payment Currency"].cat.categories).union(df["Receiving Currency"].cat.categories)
    pay_cur_codes = df["Payment Currency"].cat.set_categories(all_currencies).cat.codes.to_numpy()
    recv_cur_codes = df["Receiving Currency"].cat.set_categories(all_currencies).cat.codes.to_numpy()
    same_currency = (pay_cur_codes == recv_cur_codes).astype(np.float32)

    log_paid = np.log1p(amount_paid)
    log_received = np.log1p(amount_received)
    amount_ratio = np.divide(
        amount_received, amount_paid, out=np.ones_like(amount_paid), where=amount_paid != 0,
    ).clip(0, 5)
    hour = df["Timestamp"].dt.hour.to_numpy()
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    day_of_week = df["Timestamp"].dt.dayofweek.to_numpy(dtype=np.float32)

    node_x_numeric = np.stack(
        [log_paid, log_received, same_currency, amount_ratio, hour_sin, hour_cos, day_of_week], axis=1,
    ).astype(np.float32)

    payment_format_codes, payment_format_cats = pd.factorize(df["Payment Format"])
    pay_currency_codes, pay_currency_cats = pd.factorize(df["Payment Currency"])
    recv_currency_codes, recv_currency_cats = pd.factorize(df["Receiving Currency"])
    node_x_cat = np.stack([payment_format_codes, pay_currency_codes, recv_currency_codes], axis=1)

    node_label = df["Is Laundering"].to_numpy(dtype=np.float32)

    del df, acct_codes, bank_codes, pay_cur_codes, recv_cur_codes, all_currencies
    del payment_format_codes, pay_currency_codes, recv_currency_codes
    gc.collect()

    # Temporal train/val/test split (60/20/20)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    split_np = np.empty(n, dtype=np.int64)
    split_np[:train_end], split_np[train_end:val_end], split_np[val_end:] = 0, 1, 2
    print("Split sizes:", (split_np == 0).sum(), (split_np == 1).sum(), (split_np == 2).sum())
    for name, s in zip(["train", "val", "test"], [0, 1, 2]):
        print(f"  {name} laundering rate: {node_label[split_np == s].mean():.4%}")

    # Flow-through edges: deposit (this txn's destination account) ->
    # withdrawal (a later txn's source account, same account, within
    # TIME_WINDOW_SECONDS, capped at MAX_FANOUT nearest-in-time matches).
    print(f"\nBuilding flow-through edges (window={TIME_WINDOW_SECONDS/86400:.0f}d, "
          f"max_fanout={MAX_FANOUT})...")
    dep_order = np.lexsort((ts, deposit_key))
    wd_order = np.lexsort((ts, withdrawal_key))
    dep_key_sorted, dep_t_sorted, dep_id_sorted = deposit_key[dep_order], ts[dep_order], dep_order.astype(np.int64)
    wd_key_sorted, wd_t_sorted, wd_id_sorted = withdrawal_key[wd_order], ts[wd_order], wd_order.astype(np.int64)

    wd_boundaries = np.flatnonzero(np.diff(wd_key_sorted, prepend=wd_key_sorted[0] - 1))
    wd_boundaries = np.append(wd_boundaries, len(wd_key_sorted)).astype(np.int64)
    wd_group_keys = wd_key_sorted[wd_boundaries[:-1]]

    t0 = time.time()
    exact_edge_count = count_flow_through_edges(
        dep_key_sorted, dep_t_sorted, wd_key_sorted, wd_t_sorted,
        wd_group_keys, wd_boundaries, TIME_WINDOW_SECONDS, MAX_FANOUT,
    )
    edge_src, edge_dst, edge_gap = match_flow_through_edges(
        dep_key_sorted, dep_t_sorted, dep_id_sorted,
        wd_key_sorted, wd_t_sorted, wd_id_sorted, wd_group_keys, wd_boundaries,
        TIME_WINDOW_SECONDS, MAX_FANOUT, exact_edge_count,
    )
    print(f"  {len(edge_src):,} edges built in {time.time() - t0:.1f}s")

    out_deg = np.bincount(edge_src, minlength=n)
    in_deg = np.bincount(edge_dst, minlength=n)
    print(f"  out-degree: mean {out_deg.mean():.2f}, max {out_deg.max()}, "
          f"nodes with 0 out-edges: {(out_deg == 0).mean():.2%}")
    print(f"  in-degree:  mean {in_deg.mean():.2f}, max {in_deg.max()}, "
          f"nodes with 0 in-edges: {(in_deg == 0).mean():.2%}")
    del dep_order, wd_order, dep_key_sorted, dep_t_sorted, dep_id_sorted
    del wd_key_sorted, wd_t_sorted, wd_id_sorted, wd_boundaries, wd_group_keys
    del out_deg, in_deg, deposit_key, withdrawal_key
    gc.collect()

    # temporal check
    violations = (ts[edge_dst] <= ts[edge_src]).sum()
    assert violations == 0, f"{violations} edges violate forward-time-only construction"
    print("  every edge points forward in time")

    # Edge features
    log_gap = np.log1p(edge_gap).astype(np.float32)
    amount_match_ratio = np.divide(
        amount_paid[edge_dst], amount_received[edge_src],
        out=np.ones(len(edge_src), dtype=np.float32),
        where=amount_received[edge_src] != 0,
    ).clip(0, 5).astype(np.float32)
    edge_attr = np.stack([log_gap, amount_match_ratio], axis=1).astype(np.float32)

    torch.save({
        "num_nodes": n,
        "node_x_numeric": torch.tensor(node_x_numeric),
        "node_x_cat": torch.tensor(node_x_cat, dtype=torch.long),
        "node_label": torch.tensor(node_label),
        "split": torch.tensor(split_np),
        "edge_index": torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr),
        "cat_cardinalities": {
            "payment_format": len(payment_format_cats),
            "payment_currency": len(pay_currency_cats),
            "receiving_currency": len(recv_currency_cats),
        },
    }, OUT_PATH)
    print(f"\nSaved transaction-node graph to {OUT_PATH}")


if __name__ == "__main__":
    main()

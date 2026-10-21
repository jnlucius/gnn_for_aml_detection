import numpy as np
import pandas as pd

DTYPES = {
    "From Bank": "category", "Account": "category", "To Bank": "category", "Account.1": "category",
    "Amount Received": "float32", "Receiving Currency": "category",
    "Amount Paid": "float32", "Payment Currency": "category", "Payment Format": "category",
    "Is Laundering": "int8",
}

ACCOUNT_COLS = [
    "out_count", "in_count", "out_sum", "in_sum", "out_mean", "in_mean",
    "out_unique_cp", "in_unique_cp", "out_max_ratio", "in_max_ratio",
]


def load_transactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=DTYPES, parse_dates=["Timestamp"])  # pyright: ignore[reportArgumentType]
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    return df


def build_account_index(df: pd.DataFrame):
    n_edges = len(df)
    acct_codes, acct_cats = pd.factorize(pd.concat([df["Account"], df["Account.1"]], ignore_index=True))
    acct_src, acct_dst = acct_codes[:n_edges], acct_codes[n_edges:]
    bank_codes, bank_cats = pd.factorize(pd.concat([df["From Bank"], df["To Bank"]], ignore_index=True))
    bank_src, bank_dst = bank_codes[:n_edges], bank_codes[n_edges:]
    num_accounts_local = len(acct_cats)
    node_key_src = bank_src.astype(np.int64) * num_accounts_local + acct_src.astype(np.int64)
    node_key_dst = bank_dst.astype(np.int64) * num_accounts_local + acct_dst.astype(np.int64)
    codes, uniques = pd.factorize(np.concatenate([node_key_src, node_key_dst]))
    num_nodes = len(uniques)
    src_idx, dst_idx = codes[:n_edges], codes[n_edges:]
    return src_idx, dst_idx, num_nodes


def temporal_split(n_edges: int) -> np.ndarray:
    train_end, val_end = int(n_edges * 0.6), int(n_edges * 0.8)
    split = np.empty(n_edges, dtype=np.int64)
    split[:train_end], split[train_end:val_end], split[val_end:] = 0, 1, 2
    return split


def compute_account_features(df, src_idx, dst_idx, num_nodes, train_mask):
    amount_paid = df["Amount Paid"].to_numpy(dtype=np.float32)
    amount_received = df["Amount Received"].to_numpy(dtype=np.float32)
    train_src, train_dst = src_idx[train_mask], dst_idx[train_mask]
    train_amt_paid, train_amt_recv = amount_paid[train_mask], amount_received[train_mask]

    out_count = np.bincount(train_src, minlength=num_nodes).astype(np.float32)
    in_count = np.bincount(train_dst, minlength=num_nodes).astype(np.float32)
    out_sum = np.bincount(train_src, weights=train_amt_paid, minlength=num_nodes)
    in_sum = np.bincount(train_dst, weights=train_amt_recv, minlength=num_nodes)
    out_mean = np.divide(out_sum, out_count, out=np.zeros(num_nodes), where=out_count > 0)
    in_mean = np.divide(in_sum, in_count, out=np.zeros(num_nodes), where=in_count > 0)

    key_mod = num_nodes + 1
    out_pairs = np.unique(train_src.astype(np.int64) * key_mod + train_dst.astype(np.int64))
    out_unique_cp = np.bincount(out_pairs // key_mod, minlength=num_nodes).astype(np.float32)
    in_pairs = np.unique(train_dst.astype(np.int64) * key_mod + train_src.astype(np.int64))
    in_unique_cp = np.bincount(in_pairs // key_mod, minlength=num_nodes).astype(np.float32)

    out_max = np.zeros(num_nodes, dtype=np.float32)
    np.maximum.at(out_max, train_src, train_amt_paid)
    in_max = np.zeros(num_nodes, dtype=np.float32)
    np.maximum.at(in_max, train_dst, train_amt_recv)
    out_max_ratio = np.divide(out_max, out_mean, out=np.zeros(num_nodes), where=out_mean > 0).clip(0, 20)
    in_max_ratio = np.divide(in_max, in_mean, out=np.zeros(num_nodes), where=in_mean > 0).clip(0, 20)

    account_feats = np.stack([
        np.log1p(out_count), np.log1p(in_count), np.log1p(out_sum), np.log1p(in_sum),
        np.log1p(out_mean), np.log1p(in_mean), np.log1p(out_unique_cp), np.log1p(in_unique_cp),
        out_max_ratio, in_max_ratio,
    ], axis=1).astype(np.float32)
    return account_feats


def assemble_feature_table(df, src_idx, dst_idx, account_feats) -> pd.DataFrame:
    amount_paid = df["Amount Paid"].to_numpy(dtype=np.float32)
    amount_received = df["Amount Received"].to_numpy(dtype=np.float32)
    same_currency = (df["Payment Currency"].cat.codes.to_numpy() ==
                      df["Receiving Currency"].cat.codes.to_numpy()).astype(np.float32)
    amount_ratio = np.divide(
        amount_received, amount_paid, out=np.ones_like(amount_paid), where=amount_paid != 0,
    ).clip(0, 5)
    hour = df["Timestamp"].dt.hour.to_numpy()
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    dow = df["Timestamp"].dt.dayofweek.to_numpy(dtype=np.float32)

    feat_df = pd.DataFrame({
        "log_amount_paid": np.log1p(amount_paid),
        "log_amount_received": np.log1p(amount_received),
        "same_currency": same_currency,
        "amount_ratio": amount_ratio,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week": dow,
        "payment_format": df["Payment Format"].to_numpy(),
    })
    for i, col in enumerate(ACCOUNT_COLS):
        feat_df[f"src_{col}"] = account_feats[src_idx, i]
        feat_df[f"dst_{col}"] = account_feats[dst_idx, i]
    feat_df["payment_format"] = feat_df["payment_format"].astype("category")
    return feat_df


def build_all(csv_path: str) -> dict:
    df = load_transactions(csv_path)
    n_edges = len(df)
    src_idx, dst_idx, num_nodes = build_account_index(df)
    split = temporal_split(n_edges)
    train_mask = split == 0
    account_feats = compute_account_features(df, src_idx, dst_idx, num_nodes, train_mask)
    feat_df = assemble_feature_table(df, src_idx, dst_idx, account_feats)
    label = df["Is Laundering"].to_numpy(dtype=np.int64)
    return {
        "df": df, "feat_df": feat_df, "label": label, "split": split,
        "n_edges": n_edges, "num_nodes": num_nodes, "src_idx": src_idx, "dst_idx": dst_idx,
    }

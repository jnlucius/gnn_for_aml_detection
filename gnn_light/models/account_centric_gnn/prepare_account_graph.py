import numpy as np
import pandas as pd
import torch
import gc

SIZE = "Medium"
CSV_PATH = f"./datasets/ibm_aml/HI-{SIZE}_Trans.csv"
OUT_PATH = f"./output/aml_{SIZE}_graph_homo_v2.pt"

DTYPES = {
    "From Bank": "category",
    "Account": "category",
    "To Bank": "category",
    "Account.1": "category",
    "Amount Received": "float32",
    "Receiving Currency": "category",
    "Amount Paid": "float32",
    "Payment Currency": "category",
    "Payment Format": "category",
    "Is Laundering": "int8",
}

print("Loading CSV...")
df = pd.read_csv(CSV_PATH, dtype=DTYPES, parse_dates=["Timestamp"]) # pyright: ignore[reportArgumentType]
df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
n_edges = len(df)
print(f"{n_edges:,} transactions, "
      f"{df['Is Laundering'].mean():.4%} laundering rate")
print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")


# Global account node index
acct_codes, acct_cats = pd.factorize(pd.concat([df["Account"], df["Account.1"]], ignore_index=True))
acct_src, acct_dst = acct_codes[:n_edges], acct_codes[n_edges:]

bank_codes, bank_cats = pd.factorize(pd.concat([df["From Bank"], df["To Bank"]], ignore_index=True))
bank_src, bank_dst = bank_codes[:n_edges], bank_codes[n_edges:]

num_accounts_local = len(acct_cats)
node_key_src = bank_src.astype(np.int64) * num_accounts_local + acct_src.astype(np.int64)
node_key_dst = bank_dst.astype(np.int64) * num_accounts_local + acct_dst.astype(np.int64)
codes, uniques = pd.factorize(np.concatenate([node_key_src, node_key_dst]))
num_nodes = len(uniques)

src_idx = torch.tensor(codes[:n_edges], dtype=torch.long)
dst_idx = torch.tensor(codes[n_edges:], dtype=torch.long)
edge_index = torch.stack([src_idx, dst_idx], dim=0)
print(f"{num_nodes:,} unique accounts, {n_edges:,} directed edges")

del acct_codes, acct_cats, bank_codes, bank_cats, node_key_src, node_key_dst, uniques
gc.collect()


# Edge features
amount_paid = df["Amount Paid"].to_numpy(dtype=np.float32)
amount_received = df["Amount Received"].to_numpy(dtype=np.float32)

all_currencies = pd.Index(df["Payment Currency"].cat.categories).union(df["Receiving Currency"].cat.categories)
pay_cur_codes = df["Payment Currency"].cat.set_categories(all_currencies).cat.codes.to_numpy()
recv_cur_codes = df["Receiving Currency"].cat.set_categories(all_currencies).cat.codes.to_numpy()
same_currency = (pay_cur_codes == recv_cur_codes).astype(np.float32)

log_paid = np.log1p(amount_paid)
log_received = np.log1p(amount_received)
amount_ratio = np.divide(
    amount_received, amount_paid,
    out=np.ones_like(amount_paid), where=amount_paid != 0,
).clip(0, 5)

hour = df["Timestamp"].dt.hour.to_numpy()
hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
day_of_week = df["Timestamp"].dt.dayofweek.to_numpy(dtype=np.float32)

numeric_edge_feats = np.stack(
    [log_paid, log_received, same_currency, amount_ratio, hour_sin, hour_cos, day_of_week],
    axis=1,
).astype(np.float32)

payment_format_codes, payment_format_cats = pd.factorize(df["Payment Format"])
pay_currency_codes, pay_currency_cats = pd.factorize(df["Payment Currency"])
recv_currency_codes, recv_currency_cats = pd.factorize(df["Receiving Currency"])

edge_attr_numeric = torch.tensor(numeric_edge_feats)
edge_attr_cat = torch.tensor(
    np.stack([payment_format_codes, pay_currency_codes, recv_currency_codes], axis=1),
    dtype=torch.long,
)
edge_label = torch.tensor(df["Is Laundering"].to_numpy(dtype=np.float32))

del df, pay_cur_codes, recv_cur_codes, all_currencies
gc.collect()

# Temporal train/val/test split (60/20/20)
n = n_edges
train_end = int(n * 0.6)
val_end = int(n * 0.8)
split_np = np.empty(n, dtype=np.int64)
split_np[:train_end] = 0
split_np[train_end:val_end] = 1
split_np[val_end:] = 2
split = torch.tensor(split_np)

print("Split sizes:", (split == 0).sum().item(), (split == 1).sum().item(), (split == 2).sum().item())
for name, s in zip(["train", "val", "test"], [0, 1, 2]):
    rate = edge_label[split == s].mean().item()
    print(f"  {name} laundering rate: {rate:.4%}")

# Node features. Computed from TRAIN-period edges only
train_mask_np = split_np == 0
train_src = codes[:n_edges][train_mask_np]
train_dst = codes[n_edges:][train_mask_np]
train_amount_paid = amount_paid[train_mask_np]
train_amount_received = amount_received[train_mask_np]

node_out_count = np.bincount(train_src, minlength=num_nodes).astype(np.float32)
node_in_count = np.bincount(train_dst, minlength=num_nodes).astype(np.float32)
node_out_sum = np.bincount(train_src, weights=train_amount_paid, minlength=num_nodes)
node_in_sum = np.bincount(train_dst, weights=train_amount_received, minlength=num_nodes)
node_out_mean = np.divide(node_out_sum, node_out_count, out=np.zeros(num_nodes), where=node_out_count > 0)
node_in_mean = np.divide(node_in_sum, node_in_count, out=np.zeros(num_nodes), where=node_in_count > 0)

key_mod = num_nodes + 1
out_pair_keys = np.unique(train_src.astype(np.int64) * key_mod + train_dst.astype(np.int64))
node_out_unique_cp = np.bincount(out_pair_keys // key_mod, minlength=num_nodes).astype(np.float32)
in_pair_keys = np.unique(train_dst.astype(np.int64) * key_mod + train_src.astype(np.int64))
node_in_unique_cp = np.bincount(in_pair_keys // key_mod, minlength=num_nodes).astype(np.float32)

node_out_max = np.zeros(num_nodes, dtype=np.float32)
np.maximum.at(node_out_max, train_src, train_amount_paid)
node_in_max = np.zeros(num_nodes, dtype=np.float32)
np.maximum.at(node_in_max, train_dst, train_amount_received)

node_out_max_ratio = np.divide(node_out_max, node_out_mean, out=np.zeros(num_nodes), where=node_out_mean > 0)
node_in_max_ratio = np.divide(node_in_max, node_in_mean, out=np.zeros(num_nodes), where=node_in_mean > 0)

node_feats = np.stack([
    np.log1p(node_out_count), np.log1p(node_in_count),
    np.log1p(node_out_sum), np.log1p(node_in_sum),
    np.log1p(node_out_mean), np.log1p(node_in_mean),
    np.log1p(node_out_unique_cp), np.log1p(node_in_unique_cp),
    node_out_max_ratio.clip(0, 20), node_in_max_ratio.clip(0, 20),
], axis=1).astype(np.float32)

train_node_mask = (node_out_count + node_in_count) > 0
train_mean = node_feats[train_node_mask].mean(axis=0)
train_std = node_feats[train_node_mask].std(axis=0)
node_feats = (node_feats - train_mean) / (train_std + 1e-6)
node_x = torch.tensor(node_feats)

cold_start = (node_out_count + node_in_count) == 0
print(f"Cold-start accounts (no train-period activity): "
      f"{cold_start.sum():,} / {num_nodes:,} ({cold_start.mean():.2%})")

# Message-passing
mp_edge_index_train = edge_index[:, split == 0]
mp_edge_index_val = edge_index[:, split == 0]
mp_edge_index_test = edge_index[:, (split == 0) | (split == 1)]

print(f"Message-passing edges train: {mp_edge_index_train.size(1):,}, "
      f"val: {mp_edge_index_val.size(1):,}, test: {mp_edge_index_test.size(1):,}")

# Port numbering
def compute_ports(ref_src, ref_dst, query_src, query_dst):
    order = np.arange(len(ref_src))
    ref_df = pd.DataFrame({"a": ref_dst, "b": ref_src, "order": order})
    first_occ = ref_df.sort_values("order").drop_duplicates(["a", "b"], keep="first").copy()
    first_occ["port"] = first_occ.groupby("a").cumcount()
    port_lookup = first_occ.set_index(["a", "b"])["port"]
    next_new_port = first_occ.groupby("a")["port"].max() + 1

    query_idx = pd.MultiIndex.from_arrays([query_dst, query_src])
    looked_up = query_idx.map(port_lookup) # pyright: ignore[reportArgumentType]
    fallback = pd.Index(query_dst).map(next_new_port).to_numpy(dtype=np.float32) # pyright: ignore[reportArgumentType]
    fallback = np.nan_to_num(fallback, nan=0.0) 
    result = np.where(pd.isna(looked_up.to_numpy(dtype=object)), fallback, looked_up.to_numpy(dtype=np.float32))
    return result.astype(np.float32)

split_ref = {0: (split_np == 0), 1: (split_np == 0), 2: ((split_np == 0) | (split_np == 1))}
in_port = np.zeros(n_edges, dtype=np.float32)
out_port = np.zeros(n_edges, dtype=np.float32)
codes_src, codes_dst = codes[:n_edges], codes[n_edges:]
for s, ref_mask in split_ref.items():
    q_mask = split_np == s
    in_port[q_mask] = compute_ports(codes_src[ref_mask], codes_dst[ref_mask], codes_src[q_mask], codes_dst[q_mask])
    out_port[q_mask] = compute_ports(codes_dst[ref_mask], codes_src[ref_mask], codes_dst[q_mask], codes_src[q_mask])

in_port = np.log1p(in_port).clip(0, 10).astype(np.float32)
out_port = np.log1p(out_port).clip(0, 10).astype(np.float32)
print(f"Port features added. in_port range [{in_port.min():.2f}, {in_port.max():.2f}], "
      f"out_port range [{out_port.min():.2f}, {out_port.max():.2f}]")

edge_attr_numeric = torch.cat([
    edge_attr_numeric, torch.tensor(in_port).unsqueeze(1), torch.tensor(out_port).unsqueeze(1),
], dim=1)

torch.save({
    "num_nodes": num_nodes,
    "node_x": node_x,
    "edge_index": edge_index,
    "mp_edge_index_train": mp_edge_index_train,
    "mp_edge_index_val": mp_edge_index_val,
    "mp_edge_index_test": mp_edge_index_test,
    "edge_attr_numeric": edge_attr_numeric,
    "edge_attr_cat": edge_attr_cat,
    "edge_label": edge_label,
    "split": split,
    "cat_cardinalities": {
        "payment_format": len(payment_format_cats),
        "payment_currency": len(pay_currency_cats),
        "receiving_currency": len(recv_currency_cats),
    },
}, OUT_PATH)
print(f"\nSaved graph tensors to {OUT_PATH}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GINEConv, BatchNorm

HIDDEN = 100
NUM_NEIGHBORS = [15, 10]
CAT_EMB_DIM = 8
NUM_GNN_LAYERS = 2
EDGE_UPDATES = True
BATCH_SIZE = 4096
NUM_WORKERS = 4
TARGET_POS_RATIO = 0.05


class GINeEdgeEncoder(nn.Module):
    def __init__(self, node_in_dim, n_numeric, cat_card, hidden=HIDDEN, num_layers=NUM_GNN_LAYERS,
                 cat_emb_dim=CAT_EMB_DIM, edge_updates=EDGE_UPDATES):
        super().__init__()
        self.n_numeric = n_numeric
        self.hidden = hidden
        self.num_layers = num_layers
        self.edge_updates = edge_updates

        self.fmt_emb = nn.Embedding(cat_card["payment_format"], cat_emb_dim)
        self.pay_cur_emb = nn.Embedding(cat_card["payment_currency"], cat_emb_dim)
        self.recv_cur_emb = nn.Embedding(cat_card["receiving_currency"], cat_emb_dim)
        edge_in_dim = n_numeric + cat_emb_dim * 3

        self.node_emb = nn.Linear(node_in_dim, hidden)
        self.edge_emb = nn.Linear(edge_in_dim, hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            if edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                ))
            self.bns.append(BatchNorm(hidden))

    def _embed_edge_attr(self, raw_edge_attr):
        numeric = raw_edge_attr[:, :self.n_numeric]
        cat = raw_edge_attr[:, self.n_numeric:].long()
        fmt = self.fmt_emb(cat[:, 0])
        pay_cur = self.pay_cur_emb(cat[:, 1])
        recv_cur = self.recv_cur_emb(cat[:, 2])
        return self.edge_emb(torch.cat([numeric, fmt, pay_cur, recv_cur], dim=-1))

    def forward(self, x, edge_index, raw_edge_attr):
        src, dst = edge_index
        x = self.node_emb(x)
        edge_attr = self._embed_edge_attr(raw_edge_attr)

        for i in range(self.num_layers):
            x = (x + F.relu(self.bns[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
        return x


class EdgeClassifier(nn.Module):
    def __init__(self, node_hidden, n_numeric, cat_card, cat_emb_dim=CAT_EMB_DIM):
        super().__init__()
        self.fmt_emb = nn.Embedding(cat_card["payment_format"], cat_emb_dim)
        self.pay_cur_emb = nn.Embedding(cat_card["payment_currency"], cat_emb_dim)
        self.recv_cur_emb = nn.Embedding(cat_card["receiving_currency"], cat_emb_dim)
        in_dim = node_hidden * 2 + n_numeric + cat_emb_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h_src, h_dst, numeric_feats, cat_feats):
        fmt = self.fmt_emb(cat_feats[:, 0])
        pay_cur = self.pay_cur_emb(cat_feats[:, 1])
        recv_cur = self.recv_cur_emb(cat_feats[:, 2])
        x = torch.cat([h_src, h_dst, numeric_feats, fmt, pay_cur, recv_cur], dim=1)
        return self.mlp(x).squeeze(1)


def oversampled_seed_index(y, target_ratio):
    pos_idx = (y == 1).nonzero(as_tuple=True)[0]
    neg_idx = (y == 0).nonzero(as_tuple=True)[0]
    num_pos, num_neg = pos_idx.numel(), neg_idx.numel()
    desired_pos = int(target_ratio / (1 - target_ratio) * num_neg)
    repeat_factor = max(1, -(-desired_pos // num_pos))
    pos_idx_oversampled = pos_idx.repeat(repeat_factor)[:desired_pos]
    seed_idx = torch.cat([neg_idx, pos_idx_oversampled])
    effective_ratio = pos_idx_oversampled.numel() / seed_idx.numel()
    return seed_idx, effective_ratio, num_pos, num_neg, repeat_factor


def load_graph(graph_path):
    return torch.load(graph_path, weights_only=False)


def build_split_view(g, split_name, target_pos_ratio=TARGET_POS_RATIO):
    num_nodes = g["num_nodes"]
    node_x = g["node_x"]
    edge_index = g["edge_index"]
    mp_edge_index = {"train": g["mp_edge_index_train"], "val": g["mp_edge_index_val"], "test": g["mp_edge_index_test"]}
    edge_attr_numeric = g["edge_attr_numeric"]
    edge_attr_cat = g["edge_attr_cat"]
    edge_label = g["edge_label"]
    split = g["split"]
    N_NUMERIC = edge_attr_numeric.size(1)

    mp_mask = {"train": split == 0, "val": split == 0, "test": (split == 0) | (split == 1)}
    full_edge_attr = torch.cat([edge_attr_numeric, edge_attr_cat.float()], dim=1)
    packed_label = torch.cat([edge_label.unsqueeze(1), edge_attr_numeric, edge_attr_cat.float()], dim=1)
    global_positions = torch.arange(edge_index.size(1), dtype=torch.long)

    s = {"train": 0, "val": 1, "test": 2}[split_name]
    label_mask = split == s
    data = Data(
        x=node_x, edge_index=mp_edge_index[split_name],
        edge_attr=full_edge_attr[mp_mask[split_name]], num_nodes=num_nodes,
    )
    if split_name == "train":
        seed_idx, effective_ratio, num_pos, num_neg, repeat_factor = oversampled_seed_index(
            edge_label[label_mask], target_pos_ratio,
        )
        edge_label_index = edge_index[:, label_mask][:, seed_idx]
        edge_label_packed = packed_label[label_mask][seed_idx]
        pos_lookup = global_positions[label_mask][seed_idx]
        oversample_info = (effective_ratio, num_pos, num_neg, repeat_factor)
    else:
        edge_label_index = edge_index[:, label_mask]
        edge_label_packed = packed_label[label_mask]
        pos_lookup = global_positions[label_mask]
        oversample_info = None

    return {
        "data": data, "edge_label_index": edge_label_index, "edge_label_packed": edge_label_packed,
        "pos_lookup": pos_lookup, "N_NUMERIC": N_NUMERIC, "cat_card": g["cat_cardinalities"],
        "oversample_info": oversample_info,
    }


def make_loader(view, *, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
    return LinkNeighborLoader(
        view["data"], num_neighbors=NUM_NEIGHBORS,
        edge_label_index=view["edge_label_index"], edge_label=view["edge_label_packed"],
        batch_size=batch_size, shuffle=shuffle, neg_sampling_ratio=0.0,
        num_workers=num_workers, persistent_workers=shuffle and num_workers > 0,
    )


def subset_view(view, positions):
    pos_lookup = view["pos_lookup"]
    wanted = torch.as_tensor(positions, dtype=torch.long)
    index_of = {int(p.item()): i for i, p in enumerate(pos_lookup)}
    rows = torch.tensor([index_of[int(p)] for p in wanted], dtype=torch.long)
    return {
        "data": view["data"],
        "edge_label_index": view["edge_label_index"][:, rows],
        "edge_label_packed": view["edge_label_packed"][rows],
        "pos_lookup": view["pos_lookup"][rows],
        "N_NUMERIC": view["N_NUMERIC"], "cat_card": view["cat_card"], "oversample_info": None,
    }

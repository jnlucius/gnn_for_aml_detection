import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GINEConv, BatchNorm

HIDDEN = 100
NUM_NEIGHBORS = [15, 10]
CAT_EMB_DIM = 8
NUM_GNN_LAYERS = 2
EDGE_UPDATES = True
BATCH_SIZE = 4096
NUM_WORKERS = 4
TARGET_POS_RATIO = 0.05


class GINeNodeEncoder(nn.Module):
    def __init__(self, n_node_numeric, cat_card, hidden=HIDDEN, num_layers=NUM_GNN_LAYERS,
                 cat_emb_dim=CAT_EMB_DIM, edge_updates=EDGE_UPDATES):
        super().__init__()
        self.n_node_numeric = n_node_numeric
        self.hidden = hidden
        self.num_layers = num_layers
        self.edge_updates = edge_updates

        self.fmt_emb = nn.Embedding(cat_card["payment_format"], cat_emb_dim)
        self.pay_cur_emb = nn.Embedding(cat_card["payment_currency"], cat_emb_dim)
        self.recv_cur_emb = nn.Embedding(cat_card["receiving_currency"], cat_emb_dim)
        node_in_dim = n_node_numeric + cat_emb_dim * 3

        self.node_emb = nn.Linear(node_in_dim, hidden)
        self.edge_emb = nn.Linear(2, hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            if edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                ))
            self.bns.append(BatchNorm(hidden))

    def _embed_node_x(self, numeric, cat):
        fmt = self.fmt_emb(cat[:, 0])
        pay_cur = self.pay_cur_emb(cat[:, 1])
        recv_cur = self.recv_cur_emb(cat[:, 2])
        return self.node_emb(torch.cat([numeric, fmt, pay_cur, recv_cur], dim=-1))

    def forward(self, numeric, cat, edge_index, raw_edge_attr):
        src, dst = edge_index
        x = self._embed_node_x(numeric, cat)
        edge_attr = self.edge_emb(raw_edge_attr)

        for i in range(self.num_layers):
            x = (x + F.relu(self.bns[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
        return x


class NodeClassifier(nn.Module):
    def __init__(self, node_hidden, n_node_numeric, cat_card, cat_emb_dim=CAT_EMB_DIM):
        super().__init__()
        self.fmt_emb = nn.Embedding(cat_card["payment_format"], cat_emb_dim)
        self.pay_cur_emb = nn.Embedding(cat_card["payment_currency"], cat_emb_dim)
        self.recv_cur_emb = nn.Embedding(cat_card["receiving_currency"], cat_emb_dim)
        in_dim = node_hidden + n_node_numeric + cat_emb_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h, numeric, cat):
        fmt = self.fmt_emb(cat[:, 0])
        pay_cur = self.pay_cur_emb(cat[:, 1])
        recv_cur = self.recv_cur_emb(cat[:, 2])
        x = torch.cat([h, numeric, fmt, pay_cur, recv_cur], dim=1)
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


def build_split_view(g, split_name, target_pos_ratio=TARGET_POS_RATIO, data=None):
    num_nodes = g["num_nodes"]
    split = g["split"]
    node_label = g["node_label"]
    N_NUMERIC = g["node_x_numeric"].size(1)
    global_positions = torch.arange(num_nodes, dtype=torch.long)

    if data is None:
        data = Data(
            node_x_numeric=g["node_x_numeric"], node_x_cat=g["node_x_cat"], node_label=node_label,
            edge_index=g["edge_index"], edge_attr=g["edge_attr"], num_nodes=num_nodes,
        )

    s = {"train": 0, "val": 1, "test": 2}[split_name]
    split_mask = split == s
    if split_name == "train":
        seed_idx, effective_ratio, num_pos, num_neg, repeat_factor = oversampled_seed_index(
            node_label[split_mask], target_pos_ratio,
        )
        input_nodes = torch.where(split_mask)[0][seed_idx]
        pos_lookup = global_positions[split_mask][seed_idx]
        oversample_info = (effective_ratio, num_pos, num_neg, repeat_factor)
    else:
        input_nodes = torch.where(split_mask)[0]
        pos_lookup = global_positions[split_mask]
        oversample_info = None

    return {
        "data": data, "input_nodes": input_nodes, "pos_lookup": pos_lookup,
        "N_NUMERIC": N_NUMERIC, "cat_card": g["cat_cardinalities"], "oversample_info": oversample_info,
    }


def make_loader(view, *, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
    return NeighborLoader(
        view["data"], num_neighbors=NUM_NEIGHBORS, input_nodes=view["input_nodes"],
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, persistent_workers=shuffle and num_workers > 0,
    )


def subset_view(view, positions):
    pos_lookup = view["pos_lookup"]
    wanted = torch.as_tensor(positions, dtype=torch.long)
    index_of = {int(p.item()): i for i, p in enumerate(pos_lookup)}
    rows = torch.tensor([index_of[int(p)] for p in wanted], dtype=torch.long)
    return {
        "data": view["data"],
        "input_nodes": view["input_nodes"][rows],
        "pos_lookup": view["pos_lookup"][rows],
        "N_NUMERIC": view["N_NUMERIC"], "cat_card": view["cat_card"], "oversample_info": None,
    }

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb

from models.xgboost import features as xgb_features


def _save_retrain_log(path: str, log: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def _load_retrain_log(path: str):
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)
        return log, (log[-1]["version"] if log else 1)
    return [], 1


class XGBoostLiveScorer:
    def __init__(self, csv_path: str, checkpoint_path: str, retrain_rounds: int = 20):
        self.csv_path = csv_path
        self.retrain_rounds = retrain_rounds

        data = xgb_features.build_all(csv_path)
        self.feat_df = data["feat_df"]
        self.label = data["label"]
        self.split = data["split"]

        self.test_positions = np.where(self.split == 2)[0]

        self.booster = xgb.Booster()
        self.booster.load_model(checkpoint_path)

        self.version = 1
        self.retrain_log = []

    @property
    def n_test(self) -> int:
        return len(self.test_positions)

    def _dmatrix(self, positions) -> xgb.DMatrix:
        X = self.feat_df.iloc[positions]
        return xgb.DMatrix(X, enable_categorical=True)

    def score_positions(self, positions):
        dmat = self._dmatrix(positions)
        t0 = time.perf_counter()
        probs = self.booster.predict(dmat)
        elapsed = time.perf_counter() - t0
        return np.asarray(probs, dtype=np.float32), elapsed

    def get_targets(self, positions):
        return self.label[np.asarray(positions)]

    def retrain_on(self, positions, logger=None):
        positions = np.asarray(positions)
        X_new = self.feat_df.iloc[positions]
        y_new = self.label[positions]
        dtrain_new = xgb.DMatrix(X_new, label=y_new, enable_categorical=True)

        pos_rate = max(y_new.mean(), 1e-6)
        scale_pos_weight = ((1 - pos_rate) / pos_rate) ** 0.5

        t0 = time.perf_counter()
        self.booster = xgb.train(
            {
                "max_depth": 6, "eta": 0.05, "objective": "binary:logistic",
                "eval_metric": "aucpr", "scale_pos_weight": float(scale_pos_weight),
                "tree_method": "hist",
            },
            dtrain_new, num_boost_round=self.retrain_rounds, xgb_model=self.booster,
        )
        elapsed = time.perf_counter() - t0

        self.version += 1
        entry = {
            "version": self.version, "n_new_rows": int(len(positions)),
            "n_new_positive": int(y_new.sum()), "retrain_seconds": elapsed,
            "rounds_added": self.retrain_rounds,
        }
        self.retrain_log.append(entry)
        if logger:
            logger.info(f"Incremental retrain -> v{self.version}: {len(positions)} new rows "
                        f"({int(y_new.sum())} positive), +{self.retrain_rounds} rounds, {elapsed:.2f}s")
        return entry

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.booster.save_model(path)

    def save_retrain_log(self, path: str):
        _save_retrain_log(path, self.retrain_log)

    def load_retrain_log(self, path: str):
        self.retrain_log, self.version = _load_retrain_log(path)


class GINEAccountCentricLiveScorer:
    def __init__(self, graph_path: str, checkpoint_path: str, device: str = "cpu"):
        from models.account_centric_gnn import model as gmodel
        self.gmodel = gmodel
        self.device = torch.device(device)

        g = gmodel.load_graph(graph_path)
        self.test_view = gmodel.build_split_view(g, "test")
        self.N_NUMERIC = self.test_view["N_NUMERIC"]
        node_in_dim = self.test_view["data"].x.size(1)

        self.encoder = gmodel.GINeEdgeEncoder(node_in_dim, self.N_NUMERIC, self.test_view["cat_card"]).to(self.device)
        self.classifier = gmodel.EdgeClassifier(gmodel.HIDDEN, self.N_NUMERIC, self.test_view["cat_card"]).to(self.device)
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.classifier.load_state_dict(ckpt["classifier"])

        order = torch.argsort(self.test_view["pos_lookup"])
        self.test_positions = self.test_view["pos_lookup"][order].tolist()

        self.version = 1
        self.retrain_log = []

    @property
    def n_test(self) -> int:
        return len(self.test_positions)

    def _forward(self, batch):
        packed = batch.edge_label
        target = packed[:, 0]
        numeric_feats = packed[:, 1:1 + self.N_NUMERIC]
        cat_feats = packed[:, 1 + self.N_NUMERIC:].long()
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        src, dst = batch.edge_label_index
        logits = self.classifier(h[src], h[dst], numeric_feats, cat_feats)
        return logits, target

    def score_positions(self, positions):
        t0 = time.perf_counter()
        sub = self.gmodel.subset_view(self.test_view, positions)
        loader = self.gmodel.make_loader(sub, batch_size=len(positions), shuffle=False, num_workers=0)
        self.encoder.eval()
        self.classifier.eval()
        probs_chunks, pos_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                batch_input_id = batch.input_id
                batch = batch.to(self.device)
                logits, _ = self._forward(batch)
                probs_chunks.append(torch.sigmoid(logits).cpu())
                pos_chunks.append(sub["pos_lookup"][batch_input_id.cpu()])
        elapsed = time.perf_counter() - t0

        probs = torch.cat(probs_chunks).numpy()
        pos_out = torch.cat(pos_chunks).tolist()
        pos_to_i = {int(p): i for i, p in enumerate(pos_out)}
        order = [pos_to_i[int(p)] for p in positions]
        return probs[order].astype(np.float32), elapsed

    def get_targets(self, positions):
        sub = self.gmodel.subset_view(self.test_view, positions)
        return sub["edge_label_packed"][:, 0].numpy()

    def retrain_on(self, positions, logger=None, epochs=1, lr=0.0005):
        t0 = time.perf_counter()
        sub = self.gmodel.subset_view(self.test_view, positions)
        n_positive = int(sub["edge_label_packed"][:, 0].sum().item())
        loader = self.gmodel.make_loader(
            sub, batch_size=min(len(positions), self.gmodel.BATCH_SIZE), shuffle=True, num_workers=0,
        )
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, weight_decay=1e-5,
        )
        self.encoder.train()
        self.classifier.train()
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                logits, target = self._forward(batch)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                optimizer.step()
        elapsed = time.perf_counter() - t0

        self.version += 1
        entry = {
            "version": self.version, "n_new_rows": int(len(positions)), "n_new_positive": n_positive,
            "retrain_seconds": elapsed, "epochs": epochs,
        }
        self.retrain_log.append(entry)
        if logger:
            logger.info(f"Incremental retrain -> v{self.version}: {len(positions)} new rows "
                        f"({n_positive} positive), {epochs} fine-tune epoch(s), {elapsed:.2f}s")
        return entry

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"encoder": self.encoder.state_dict(), "classifier": self.classifier.state_dict()}, path)

    def save_retrain_log(self, path: str):
        _save_retrain_log(path, self.retrain_log)

    def load_retrain_log(self, path: str):
        self.retrain_log, self.version = _load_retrain_log(path)


class GINETransactionCentricLiveScorer:
    def __init__(self, graph_path: str, checkpoint_path: str, device: str = "cpu"):
        from models.transaction_centric_gnn import model as gmodel
        self.gmodel = gmodel
        self.device = torch.device(device)

        g = gmodel.load_graph(graph_path)
        self.test_view = gmodel.build_split_view(g, "test")
        self.N_NUMERIC = self.test_view["N_NUMERIC"]

        self.encoder = gmodel.GINeNodeEncoder(self.N_NUMERIC, self.test_view["cat_card"]).to(self.device)
        self.classifier = gmodel.NodeClassifier(gmodel.HIDDEN, self.N_NUMERIC, self.test_view["cat_card"]).to(self.device)
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.classifier.load_state_dict(ckpt["classifier"])

        order = torch.argsort(self.test_view["pos_lookup"])
        self.test_positions = self.test_view["pos_lookup"][order].tolist()

        self.version = 1
        self.retrain_log = []

    @property
    def n_test(self) -> int:
        return len(self.test_positions)

    def _forward(self, batch, n_seed):
        h = self.encoder(batch.node_x_numeric, batch.node_x_cat, batch.edge_index, batch.edge_attr)
        seed_h = h[:n_seed]
        logits = self.classifier(seed_h, batch.node_x_numeric[:n_seed], batch.node_x_cat[:n_seed])
        target = batch.node_label[:n_seed]
        return logits, target

    def score_positions(self, positions):
        t0 = time.perf_counter()
        sub = self.gmodel.subset_view(self.test_view, positions)
        loader = self.gmodel.make_loader(sub, batch_size=len(positions), shuffle=False, num_workers=0)
        self.encoder.eval()
        self.classifier.eval()
        probs_chunks, pos_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                batch_input_id = batch.input_id
                n_seed = batch_input_id.size(0)
                batch = batch.to(self.device)
                logits, _ = self._forward(batch, n_seed)
                probs_chunks.append(torch.sigmoid(logits).cpu())
                pos_chunks.append(sub["pos_lookup"][batch_input_id.cpu()])
        elapsed = time.perf_counter() - t0

        probs = torch.cat(probs_chunks).numpy()
        pos_out = torch.cat(pos_chunks).tolist()
        pos_to_i = {int(p): i for i, p in enumerate(pos_out)}
        order = [pos_to_i[int(p)] for p in positions]
        return probs[order].astype(np.float32), elapsed

    def get_targets(self, positions):
        return self.test_view["data"].node_label[torch.as_tensor(positions, dtype=torch.long)].numpy()

    def retrain_on(self, positions, logger=None, epochs=1, lr=0.0005):
        t0 = time.perf_counter()
        sub = self.gmodel.subset_view(self.test_view, positions)
        n_positive = int(self.test_view["data"].node_label[torch.as_tensor(positions, dtype=torch.long)].sum().item())
        loader = self.gmodel.make_loader(
            sub, batch_size=min(len(positions), self.gmodel.BATCH_SIZE), shuffle=True, num_workers=0,
        )
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, weight_decay=1e-5,
        )
        self.encoder.train()
        self.classifier.train()
        for _ in range(epochs):
            for batch in loader:
                batch_input_id = batch.input_id
                n_seed = batch_input_id.size(0)
                batch = batch.to(self.device)
                optimizer.zero_grad()
                logits, target = self._forward(batch, n_seed)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                optimizer.step()
        elapsed = time.perf_counter() - t0

        self.version += 1
        entry = {
            "version": self.version, "n_new_rows": int(len(positions)), "n_new_positive": n_positive,
            "retrain_seconds": elapsed, "epochs": epochs,
        }
        self.retrain_log.append(entry)
        if logger:
            logger.info(f"Incremental retrain -> v{self.version}: {len(positions)} new rows "
                        f"({n_positive} positive), {epochs} fine-tune epoch(s), {elapsed:.2f}s")
        return entry

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"encoder": self.encoder.state_dict(), "classifier": self.classifier.state_dict()}, path)

    def save_retrain_log(self, path: str):
        _save_retrain_log(path, self.retrain_log)

    def load_retrain_log(self, path: str):
        self.retrain_log, self.version = _load_retrain_log(path)

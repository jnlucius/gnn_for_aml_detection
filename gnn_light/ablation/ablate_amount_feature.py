import torch

SIZE = "Medium"
IN_PATH = f"./output/aml_{SIZE}_graph_txnode_v1.pt"
OUT_PATH = f"./output/aml_{SIZE}_graph_txnode_v1_NOAMOUNT.pt"

g = torch.load(IN_PATH, weights_only=False)
edge_attr = g["edge_attr"].clone()
print("edge_attr columns: [log_gap, amount_match_ratio]")
print(f"amount_match_ratio before: mean={edge_attr[:,1].mean():.3f}, std={edge_attr[:,1].std():.3f}")

edge_attr[:, 1] = 0.0
g["edge_attr"] = edge_attr

torch.save(g, OUT_PATH)
print(f"Saved amount_match_ratio-ablated graph to {OUT_PATH}")

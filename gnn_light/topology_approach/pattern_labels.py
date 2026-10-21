import re
import pandas as pd
import torch

SIZE = "Medium"
CSV_PATH = f"./datasets/ibm_aml/HI-{SIZE}_Trans.csv"
PATTERNS_PATH = f"./datasets/ibm_aml/HI-{SIZE}_Patterns.txt"
OUT_PATH = f"./output/aml_{SIZE}_pattern_labels.pt"

COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]

DTYPES = {
    "From Bank": "category", "Account": "category", "To Bank": "category", "Account.1": "category",
    "Amount Received": "float32", "Receiving Currency": "category",
    "Amount Paid": "float32", "Payment Currency": "category",
    "Payment Format": "category", "Is Laundering": "int8",
}

BEGIN_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")


def parse_patterns_file(path):
    rows = []
    current_pattern = None
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            m = BEGIN_RE.match(line)
            if m:
                current_pattern = m.group(1)
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                current_pattern = None
                continue
            if current_pattern is not None:
                fields = line.split(",")
                assert len(fields) == 11, f"Unexpected field count ({len(fields)}): {line}"
                rows.append(fields + [current_pattern])
    pdf = pd.DataFrame(rows, columns=COLUMNS + ["Pattern"])
    return pdf


def build_match_key_df(df):
    return pd.DataFrame({
        "ts": df["Timestamp"].astype("int64"),
        "from_bank": df["From Bank"].cat.codes.astype("int32"),
        "account": df["Account"].cat.codes.astype("int32"),
        "to_bank": df["To Bank"].cat.codes.astype("int32"),
        "account1": df["Account.1"].cat.codes.astype("int32"),
        "amt_recv_d": df["Amount Received"].astype("float64").round().astype("int64"),
        "recv_cur": df["Receiving Currency"].cat.codes.astype("int32"),
        "amt_paid_d": df["Amount Paid"].astype("float64").round().astype("int64"),
        "pay_cur": df["Payment Currency"].cat.codes.astype("int32"),
        "pay_fmt": df["Payment Format"].cat.codes.astype("int32"),
        "is_laundering": df["Is Laundering"].astype("int8"),
    })


def build_match_key_pdf(pdf, cat_categories):
    def codes_for(col_name, cats):
        return pd.Categorical(pdf[col_name], categories=cats).codes.astype("int32")

    return pd.DataFrame({
        "ts": pdf["Timestamp"].astype("int64"),
        "from_bank": codes_for("From Bank", cat_categories["From Bank"]),
        "account": codes_for("Account", cat_categories["Account"]),
        "to_bank": codes_for("To Bank", cat_categories["To Bank"]),
        "account1": codes_for("Account.1", cat_categories["Account.1"]),
        "amt_recv_d": pdf["Amount Received"].astype("float64").round().astype("int64"),
        "recv_cur": codes_for("Receiving Currency", cat_categories["Receiving Currency"]),
        "amt_paid_d": pdf["Amount Paid"].astype("float64").round().astype("int64"),
        "pay_cur": codes_for("Payment Currency", cat_categories["Payment Currency"]),
        "pay_fmt": codes_for("Payment Format", cat_categories["Payment Format"]),
        "is_laundering": pdf["Is Laundering"].astype("int8"),
    })


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, dtype=DTYPES, parse_dates=["Timestamp"])  # pyright: ignore
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    n_edges = len(df)
    print(f"{n_edges:,} transactions loaded.")

    print("Parsing Patterns.txt...")
    pdf = parse_patterns_file(PATTERNS_PATH)
    pdf["Timestamp"] = pd.to_datetime(pdf["Timestamp"])
    for col in ["Amount Received", "Amount Paid"]:
        pdf[col] = pdf[col].astype(float)
    pdf["Is Laundering"] = pdf["Is Laundering"].astype(int)
    n_pattern_rows = len(pdf)
    print(f"{n_pattern_rows:,} labeled transaction lines across "
          f"{pdf['Pattern'].nunique()} pattern types:")
    print(pdf["Pattern"].value_counts().to_string())

    print("Building match keys...")
    key_cols = ["ts", "from_bank", "account", "to_bank", "account1",
                "amt_recv_d", "recv_cur", "amt_paid_d", "pay_cur", "pay_fmt", "is_laundering"]
    df_keys = build_match_key_df(df)
    cat_categories = {
        col: df[col].cat.categories
        for col in ["From Bank", "Account", "To Bank", "Account.1", "Receiving Currency", "Payment Currency", "Payment Format"]
    }
    pdf_keys = build_match_key_pdf(pdf, cat_categories)

    # occurrence counters so duplicate rows match nth-to-nth
    df_keys["orig_pos"] = df.index.values
    df_keys["occ"] = df_keys.groupby(key_cols).cumcount()
    pdf_keys["occ"] = pdf_keys.groupby(key_cols).cumcount()
    pdf_keys["pattern"] = pdf["Pattern"].values

    merged = pdf_keys.merge(df_keys[key_cols + ["occ", "orig_pos"]], on=key_cols + ["occ"], how="left")
    matched_pos = merged["orig_pos"]

    n_matched = matched_pos.notna().sum()
    print(f"Matched {n_matched:,} / {n_pattern_rows:,} pattern rows to transaction positions "
          f"({n_matched / n_pattern_rows:.2%}).")
    if n_matched < n_pattern_rows:
        unmatched = merged.loc[matched_pos.isna(), "pattern"].value_counts()
        print("WARNING: some pattern rows did not match. By pattern type:")
        print(unmatched.to_string())

    pattern_of_edge = [None] * n_edges
    for pos, pattern in zip(matched_pos.dropna().astype(int), merged.loc[matched_pos.notna(), "pattern"]):
        pattern_of_edge[pos] = pattern

    n_labeled = sum(p is not None for p in pattern_of_edge)
    print(f"\n{n_labeled:,} / {n_edges:,} transactions now carry a specific pattern label "
          f"({n_labeled / n_edges:.4%}).")
    is_laundering_total = df["Is Laundering"].sum()
    print(f"For reference, {is_laundering_total:,} transactions are labeled Is Laundering=1 overall - "
          f"{n_labeled / is_laundering_total:.2%} of laundering transactions have a specific pattern type")

    torch.save({"pattern_of_edge": pattern_of_edge, "n_edges": n_edges}, OUT_PATH)
    print(f"\nSaved pattern labels to {OUT_PATH}")


if __name__ == "__main__":
    main()

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

REPO_DIR = "/opt/airflow/gnn_light"
sys.path.insert(0, REPO_DIR)

from gnn_light.common import get_logger 
from gnn_light.simulation.live_scorer import GINETransactionCentricLiveScorer

DATASET_SIZE = "Medium"
RUN_TAG = ""
GRAPH_PATH = f"{REPO_DIR}/output/aml_{DATASET_SIZE}_graph_txnode_v1.pt"
CHECKPOINT_PATH = f"{REPO_DIR}/output/aml_{DATASET_SIZE}_ckpt_txnode_gine{RUN_TAG}.pt"

BATCH_SIZE = 10 
FRAUD_THRESHOLD = 0.59 

PROCESSED_LOG = f"{REPO_DIR}/output/example_processed_transactions.jsonl"
ALERTS_LOG = f"{REPO_DIR}/output/example_fraud_alerts.jsonl"


def score_transaction_batch(**context):
    logger = get_logger("example_txn_centric_scoring", DATASET_SIZE, RUN_TAG)
    scorer = GINETransactionCentricLiveScorer(GRAPH_PATH, CHECKPOINT_PATH)

    positions = list(range(BATCH_SIZE))
    probs, elapsed = scorer.score_positions(positions)
    targets = scorer.get_targets(positions)

    results = [
        {"position": int(p), "score": float(s), "true_label": int(t)}
        for p, s, t in zip(positions, probs, targets)
    ]
    logger.info(f"Scored {len(results)} transactions in {elapsed:.3f}s "
                f"({sum(r['score'] >= FRAUD_THRESHOLD for r in results)} flagged).")

    context["ti"].xcom_push(key="scored_transactions", value=results)
    return results


def branch_on_scores(**context):
    results = context["ti"].xcom_pull(task_ids="score_transaction_batch", key="scored_transactions")
    branches = []
    if any(r["score"] < FRAUD_THRESHOLD for r in results):
        branches.append("process_legit_transactions")
    if any(r["score"] >= FRAUD_THRESHOLD for r in results):
        branches.append("raise_fraud_alerts")
    return branches


def process_legit_transactions(**context):
    results = context["ti"].xcom_pull(task_ids="score_transaction_batch", key="scored_transactions")
    legit = [r for r in results if r["score"] < FRAUD_THRESHOLD]

    with open(PROCESSED_LOG, "a") as f:
        for r in legit:
            f.write(f'{{"position": {r["position"]}, "score": {r["score"]:.4f}, '
                     f'"status": "processed"}}\n')
    print(f"Processed {len(legit)} legitimate transaction(s).")


def raise_fraud_alerts(**context):
    results = context["ti"].xcom_pull(task_ids="score_transaction_batch", key="scored_transactions")
    fraud = [r for r in results if r["score"] >= FRAUD_THRESHOLD]

    with open(ALERTS_LOG, "a") as f:
        for r in fraud:
            f.write(f'{{"position": {r["position"]}, "score": {r["score"]:.4f}, '
                     f'"status": "fraud_alert"}}\n')
            print(f"FRAUD ALERT transaction {r['position']}: score {r['score']:.4f} "
                  f"(threshold {FRAUD_THRESHOLD})")


with DAG(
    dag_id="example_transaction_centric_scoring",
    schedule=timedelta(minutes=5),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1
) as dag:

    score_batch = PythonOperator(
        task_id="score_transaction_batch",
        python_callable=score_transaction_batch,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_scores",
        python_callable=branch_on_scores,
    )

    process_legit = PythonOperator(
        task_id="process_legit_transactions",
        python_callable=process_legit_transactions,
    )

    raise_alerts = PythonOperator(
        task_id="raise_fraud_alerts",
        python_callable=raise_fraud_alerts,
    )

    batch_complete = EmptyOperator(
        task_id="batch_complete",
        trigger_rule="none_failed_min_one_success",
    )

    score_batch >> branch >> [process_legit, raise_alerts] >> batch_complete

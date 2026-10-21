import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

REPO_DIR = "/opt/airflow/gnn_light"
sys.path.insert(0, REPO_DIR)

from gnn_light.common import get_logger
from gnn_light.simulation.stream_transactions import process_next_batch, process_next_batch_online

DATASET_SIZE = "Small"
MODEL_NAME = "xgboost"
RUN_TAG = ""
BATCH_SIZE = 50
OUTPUT_DIR = f"{REPO_DIR}/output"

SCORING_MODE = "replay" 
CSV_PATH = None 
GRAPH_PATH = None
RETRAIN_EVERY = 1000 
RETRAIN_ROUNDS = 20 
RETRAIN_EPOCHS = 1

EXPECTED_ALERT_RATE_MIN = 0.005
EXPECTED_ALERT_RATE_MAX = 0.20


def score_next_batch(**context):
    logger = get_logger("fraud_stream", DATASET_SIZE, RUN_TAG)
    if SCORING_MODE == "online":
        status = process_next_batch_online(
            output_dir=OUTPUT_DIR, size=DATASET_SIZE, model_name=MODEL_NAME, run_tag=RUN_TAG,
            batch_size=BATCH_SIZE, logger=logger, csv_path=CSV_PATH, graph_path=GRAPH_PATH,
            retrain_every=RETRAIN_EVERY, retrain_rounds=RETRAIN_ROUNDS, retrain_epochs=RETRAIN_EPOCHS,
        )
    else:
        status = process_next_batch(
            output_dir=OUTPUT_DIR, size=DATASET_SIZE, model_name=MODEL_NAME,
            run_tag=RUN_TAG, batch_size=BATCH_SIZE, logger=logger,
        )
    context["ti"].xcom_push(key="batch_status", value=status)
    if status["stream_complete"]:
        logger.info("Stream fully replayed")
    return status


def check_pipeline_health(**context):
    status = context["ti"].xcom_pull(task_ids="score_next_batch", key="batch_status")
    rate = status["alert_rate"]
    if not (EXPECTED_ALERT_RATE_MIN <= rate <= EXPECTED_ALERT_RATE_MAX):
        raise ValueError(
            f"Fraud alert rate {rate:.2%} (over {status['total_processed']} txns so far) is outside the "
            f"expected band [{EXPECTED_ALERT_RATE_MIN:.2%}, {EXPECTED_ALERT_RATE_MAX:.2%}] -- "
            f"possible model drift. Investigate."
        )
    print(f"Pipeline health OK - alert rate {rate:.2%}, running precision {status['running_precision']}.")


with DAG(
    dag_id="aml_fraud_stream_simulation",
    schedule=timedelta(minutes=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1
) as dag:

    score_batch = PythonOperator(task_id="score_next_batch", python_callable=score_next_batch)
    health_check = PythonOperator(task_id="check_pipeline_health", python_callable=check_pipeline_health)

    score_batch >> health_check

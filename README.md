# About this Project
This project investigates the application of Graph Neural Networks (GNNs) to financial
transaction fraud detection system. The transaction data is converted to graph representations. The effectiveness of different representations is explored to assess their impact on fraud prediction. GNN models are trained on the data and compared against traditional machine learning models in terms of fraud classification performance.

# Instructions to Run This Project

### 1. Install dependencies

Install the Python packages listed in `requirements.txt`. **Do not install directly from this file.**

PyTorch and PyTorch Geometric must be installed manually according to your CUDA version. Some PyTorch Geometric packages are also only available on Linux.

### 2. Navigate to the project

```bash
cd gnn_light
```

### 3. Download the dataset

Download the required dataset files and place them in:

```text
gnn_light/datasets/ibm_aml/
```

### 4. Prepare the graph

Run the graph preparation script, for example:

```bash
python models/account_centric_gnn/prepare_account_graph.py   
```

Configure the dataset size in the script.

### 5. Train the model

Run the training script, for example:

```bash
python models/account_centric_gnn/train_gine.py
```

Configure the dataset size in the script.

Each run produces files such as:

```text
/output/aml_{SIZE}_{model}{TAG}_run_summary.json
/output/aml_{SIZE}_{model}{TAG}_top_flagged.json
/output/aml_{SIZE}_{model}{TAG}_pr_curve.json
/output/aml_{SIZE}_{model}{TAG}_predictions.pt
```

To run **evaluation only**, without training, set:

```python
EVAL_ONLY = True
```

in the relevant script. This requires the corresponding checkpoint file to be present.

### 6. Run the dashboard

Serve the `output` folder over HTTP:

```bash
cd output
python -m http.server 8000
```

Then open:

http://localhost:8000/dashboard/index.html

If the dashboard displays **"DEMO"** data, the required output files are missing. Generate them or download them from Google Drive.

### 7. Run the streaming simulation

From the `gnn_light` directory, run:

```bash
python -m simulation.stream_transactions --loop --model-name gine_transaction_centric --size Medium --mode online
```

You can change the model and dataset size as required.

---

## Additional Components

* **Ablation:** Graph preparation scripts are located in `/ablation`.
* **Topology evaluation:** The laundering topology breakdown scripts are in `/evaluation`. This requires the Patterns file from the IBM AML dataset. Pattern labels must be generated first:

  ```bash
  python topology_approach/pattern_labels.py
  python evaluation/pattern_eval.py
  ```

  Configure the model in the evaluation script.
* **Logs:** Previous run logs are stored in `/logs`.
* **Airflow:** DAGs are located in `/dags`. Running them requires an Airflow environment on Linux or Docker. The DAG files should then be placed in the Airflow DAGs directory.


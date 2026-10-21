const DATA_BASE_PATH = '../../';

const STATUS_URL = `${DATA_BASE_PATH}stream_status.json`;
const EVENTS_URL = `${DATA_BASE_PATH}stream_events.jsonl`;
const POLL_MS = 1800;
const LEDGER_ROWS = 14;

const MODEL_CATALOG = [
  { id: 'xgboost',                  label: 'XGBoost',                    color: '#5aa9ff' },
  { id: 'gine_account_centric',     label: 'GINE (account-centric)',     color: '#3ddc97' },
  { id: 'gine_transaction_centric', label: 'GINE (transaction-centric)', color: '#f2b84b' },
  { id: 'topology_baseline',        label: 'Topology baseline',          color: '#ff5c5c' },
];

const K_OPTIONS = [100, 500, 1000, 5000];

const DEMO_PROFILES = {
  xgboost:                  { auprc:0.142, auroc:0.951, f1:0.231, precision:0.187, recall:0.302, best_threshold:0.41, ms_per_txn:0.048, n_positive:520,  p_at_k:{100:0.62,500:0.34,1000:0.21,5000:0.06} },
  gine_account_centric:     { auprc:0.224, auroc:0.968, f1:0.312, precision:0.276, recall:0.359, best_threshold:0.37, ms_per_txn:1.84,  n_positive:520,  p_at_k:{100:0.79,500:0.51,1000:0.33,5000:0.09} },
  gine_transaction_centric: { auprc:0.196, auroc:0.961, f1:0.288, precision:0.243, recall:0.354, best_threshold:0.39, ms_per_txn:2.41,  n_positive:520,  p_at_k:{100:0.74,500:0.46,1000:0.29,5000:0.08} },
  topology_baseline:        { auprc:0.061, auroc:0.712, f1:0.098, precision:0.071, recall:0.158, best_threshold:0.52, ms_per_txn:0.006, n_positive:520,  p_at_k:{100:0.31,500:0.15,1000:0.09,5000:0.03} },
};

function fmtPct(x){ return x === null || x === undefined ? '—' : (x*100).toFixed(2) + '%'; }
function fmtNum(x){ return x === null || x === undefined ? '—' : x.toLocaleString(); }
function modelMeta(id){ return MODEL_CATALOG.find(m => m.id === id); }

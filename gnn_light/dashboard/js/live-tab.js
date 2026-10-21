let mode = null;
let chart = null;
let chartPoints = [];
let lastRecordedProcessed = null;
let demoState = null;

function setStatus(newMode){
  if(mode === newMode) return;
  mode = newMode;
  const pill = document.getElementById('statusPill');
  const text = document.getElementById('statusText');
  pill.classList.remove('live','demo');
  pill.classList.add(newMode);
  text.textContent = newMode === 'live' ? 'LIVE — reading stream_status.json' : 'DEMO MODE — simulated data (run backend)';
}

function renderStatus(status){
  document.getElementById('metaRow').innerHTML =
    `model <b>${status.model_name}</b> &nbsp;·&nbsp; dataset <b>${status.dataset_size}</b> &nbsp;·&nbsp; threshold <b>${status.threshold}</b>`;
  document.getElementById('kpiTotal').textContent = fmtNum(status.total_processed);
  document.getElementById('kpiOfTotal').textContent = fmtNum(status.total_transactions);
  document.getElementById('kpiProcessed').textContent = fmtNum(status.total_processed - status.total_alerts);
  document.getElementById('kpiAlerts').textContent = fmtNum(status.total_alerts);
  document.getElementById('kpiAlertRate').textContent = fmtPct(status.alert_rate) + ' alert rate';
  document.getElementById('kpiPrecision').textContent = fmtPct(status.running_precision);
  document.getElementById('kpiRecall').textContent = fmtPct(status.running_recall);
  document.getElementById('kpiFN').textContent = fmtNum(status.false_negatives);
  document.getElementById('statTP').textContent = fmtNum(status.true_positives);
  document.getElementById('statFP').textContent = fmtNum(status.false_positives);
  document.getElementById('statTN').textContent = fmtNum(status.true_negatives);
  document.getElementById('statFN').textContent = fmtNum(status.false_negatives);

  recordChartPoint(status);
  updateChart();
  renderScoringEngine(status);
}


function recordChartPoint(status){
  const tp = status.total_processed;
  if(lastRecordedProcessed !== null && tp < lastRecordedProcessed){
    chartPoints = [];
  }
  if(chartPoints.length === 0 || tp !== chartPoints[chartPoints.length - 1].x){
    chartPoints.push({ x: tp, processed: tp - status.total_alerts, alerts: status.total_alerts });
    if(chartPoints.length > 60) chartPoints.shift();
  }
  lastRecordedProcessed = tp;
}

function renderScoringEngine(status){
  const isOnline = status.scoring_mode === 'online';
  document.getElementById('scoringModeBadge').textContent = isOnline ? 'ONLINE' : 'REPLAY';
  document.getElementById('scoringModeBadge').className = 'mode-badge ' + (isOnline ? 'online' : 'replay');
  document.getElementById('scoringModeHint').textContent = isOnline
    ? 'Model is scoring each batch live and retraining periodically on newly streamed data.'
    : 'Replaying predictions computed once during training.';

  document.getElementById('roModelVersion').textContent = status.model_version ? `v${status.model_version}` : '—';
  document.getElementById('roScoringLatency').textContent = status.avg_scoring_ms_per_txn_live !== undefined && status.avg_scoring_ms_per_txn_live !== null
    ? `${status.avg_scoring_ms_per_txn_live.toFixed(4)} ms` : '—';
  document.getElementById('roRetrainCount').textContent = status.retrain_count !== undefined ? fmtNum(status.retrain_count) : '—';

  const untilNext = (status.retrain_every !== undefined && status.since_last_retrain !== undefined)
    ? Math.max(status.retrain_every - status.since_last_retrain, 0) : null;
  document.getElementById('roNextRetrain').textContent = untilNext !== null ? `in ${fmtNum(untilNext)} txns` : '—';

  const lr = status.last_retrain;
  document.getElementById('lastRetrainNote').textContent = lr
    ? `Last retrain -> v${lr.version}: ${fmtNum(lr.n_new_rows)} new rows (${fmtNum(lr.n_new_positive)} labeled fraud), +${lr.rounds_added} boosting rounds, ${lr.retrain_seconds.toFixed(2)}s.`
    : (isOnline ? 'No retrain yet - filling the first window.' : 'Not applicable in replay mode.');
}

function decisionBadge(ev){
  const isAlert = ev.decision === 'FRAUD_ALERT';
  const mismatch = (isAlert && ev.actual_label === 0) || (!isAlert && ev.actual_label === 1);
  let flag = '';
  if(mismatch) flag = isAlert ? '<span class="flag fp">false alarm</span>' : '<span class="flag fp">missed</span>';
  return `<span class="badge ${isAlert ? 'alert' : 'processed'}">${isAlert ? 'FRAUD ALERT' : 'PROCESSED'}</span>${flag}`;
}

function renderLedger(events){
  const body = document.getElementById('ledgerBody');
  const rows = events.slice(-LEDGER_ROWS).reverse();
  body.innerHTML = rows.map(ev => {
    const t = new Date(ev.timestamp);
    const timeStr = isNaN(t) ? '' : t.toLocaleTimeString();
    return `<tr>
      <td>#${ev.seq}</td>
      <td>${timeStr}</td>
      <td>txn_${ev.txn_pos}</td>
      <td class="score">${ev.score.toFixed(3)}</td>
      <td>${decisionBadge(ev)}</td>
    </tr>`;
  }).join('');
}

function updateChart(){
  if(!chart) return;
  try{
    chart.data.datasets[0].data = chartPoints.map(p => ({x: p.x, y: p.processed}));
    chart.data.datasets[1].data = chartPoints.map(p => ({x: p.x, y: p.alerts}));
    chart.update('none');
  }catch(e){
    console.error('volumeChart update failed', e);
    chart = null;
  }
}

function initChart(){
  try{
    const ctx = document.getElementById('volumeChart');
    chart = new Chart(ctx, {
      type: 'line',
      data: { datasets: [
        { label: 'Processed (cumulative)', data: [], borderColor: '#3ddc97', backgroundColor:'rgba(61,220,151,.08)', fill:true, tension:.25, pointRadius:0, borderWidth:2 },
        { label: 'Fraud alerts (cumulative)', data: [], borderColor: '#f2b84b', backgroundColor:'rgba(242,184,75,.12)', fill:true, tension:.25, pointRadius:0, borderWidth:2, yAxisID:'y1' },
      ]},
      options: {
        responsive:true, maintainAspectRatio:false,
        interaction:{mode:'index', intersect:false},
        plugins:{ legend:{ labels:{ color:'#7c8798', font:{family:'Inter', size:11} } } },
        scales:{
          x:{ type:'linear', title:{display:true,text:'transactions processed',color:'#4b5563',font:{size:10}}, ticks:{ color:'#4b5563', font:{size:10} }, grid:{ color:'#1f2630' } },
          y:{ min:0, ticks:{ color:'#7c8798', font:{size:10} }, grid:{ color:'#1f2630' } },
          y1:{ position:'right', min:0, ticks:{ color:'#f2b84b', font:{size:10} }, grid:{ display:false } },
        }
      }
    });
  }catch(e){
    console.error('Chart.js failed to initialize the volume chart', e);
    chart = null;
  }
}

// live mode
async function tryLive(){
  try{
    const [statusRes, eventsRes] = await Promise.all([fetch(STATUS_URL, {cache:'no-store'}), fetch(EVENTS_URL, {cache:'no-store'})]);
    if(!statusRes.ok || !eventsRes.ok) throw new Error('not found');
    const status = await statusRes.json();
    const text = await eventsRes.text();
    const events = text.trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
    setStatus('live');
    renderStatus(status);
    renderLedger(events);
    return true;
  }catch(e){
    return false;
  }
}

// demo mode
function initDemo(){
  demoState = {
    total_transactions: 4000, total_processed: 0, total_alerts: 0,
    tp:0, fp:0, tn:0, fn:0, events: [], seq: 0,
    model_version: 1, retrain_count: 0, retrain_every: 500, since_last_retrain: 0,
    total_scoring_seconds: 0, last_retrain: null,
  };
}

function demoTick(){
  const s = demoState;
  if(s.total_processed >= s.total_transactions) return;
  const batch = Math.min(9, s.total_transactions - s.total_processed);
  for(let i=0;i<batch;i++){
    const actual = Math.random() < 0.028 ? 1 : 0;
    const score = actual ? Math.min(0.99, 0.55 + Math.random()*0.45) : Math.random()*0.5;
    const threshold = 0.5;
    const isAlert = score >= threshold;
    s.seq += 1;
    const ev = { seq:s.seq, txn_pos: 100000 + s.seq, score, threshold, decision: isAlert ? 'FRAUD_ALERT':'PROCESSED', actual_label:actual, model_version: s.model_version, timestamp: new Date().toISOString() };
    s.events.push(ev);
    if(s.events.length > 200) s.events.shift();
    if(isAlert && actual===1) s.tp++; else if(isAlert && actual===0) s.fp++; else if(!isAlert && actual===1) s.fn++; else s.tn++;
    if(isAlert) s.total_alerts++;
    s.total_processed++;
    s.since_last_retrain++;
    s.total_scoring_seconds += (0.03 + Math.random()*0.02) / 1000;
  }
  if(s.since_last_retrain >= s.retrain_every){
    s.model_version++;
    s.retrain_count++;
    s.last_retrain = { version: s.model_version, n_new_rows: s.retrain_every, n_new_positive: Math.round(s.retrain_every*0.028), rounds_added: 20, retrain_seconds: 0.4 + Math.random()*0.3 };
    s.since_last_retrain = 0;
  }
  const status = {
    model_name:'xgboost (demo)', dataset_size:'Small', threshold:0.5,
    scoring_mode: 'online',
    total_transactions:s.total_transactions, total_processed:s.total_processed, total_alerts:s.total_alerts,
    alert_rate: s.total_alerts / s.total_processed,
    running_precision: (s.tp+s.fp) ? s.tp/(s.tp+s.fp) : null,
    running_recall: (s.tp+s.fn) ? s.tp/(s.tp+s.fn) : null,
    true_positives:s.tp, false_positives:s.fp, true_negatives:s.tn, false_negatives:s.fn,
    model_version: s.model_version,
    avg_scoring_ms_per_txn_live: (s.total_scoring_seconds / s.total_processed) * 1000,
    retrain_count: s.retrain_count,
    retrain_every: s.retrain_every,
    since_last_retrain: s.since_last_retrain,
    last_retrain: s.last_retrain,
  };
  setStatus('demo');
  renderStatus(status);
  renderLedger(s.events);
}

async function pollLoop(){
  const ok = await tryLive();
  if(!ok){
    if(!demoState) initDemo();
    demoTick();
  }
  setTimeout(pollLoop, POLL_MS);
}

function initLiveTab(){
  initChart();
  pollLoop();
}

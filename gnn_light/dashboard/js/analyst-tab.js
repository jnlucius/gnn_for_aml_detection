let selectedModels = new Set(['xgboost', 'gine_account_centric']);
let primaryModelId = null;
let selectedK = 500;
let cache = {};
let compareChart = null;
let thresholdChart = null;
let sliderIndex = null;

async function fetchModelResult(modelId){
  const size = document.getElementById('datasetSizeInput').value.trim() || 'Small';
  const tag = document.getElementById('runTagInput').value.trim();
  const base = `aml_${size}_${modelId}${tag}`;
  try{
    const [summaryRes, flaggedRes, curveRes] = await Promise.all([
      fetch(`${DATA_BASE_PATH}${base}_run_summary.json`, {cache:'no-store'}),
      fetch(`${DATA_BASE_PATH}${base}_top_flagged.json`, {cache:'no-store'}),
      fetch(`${DATA_BASE_PATH}${base}_pr_curve.json`, {cache:'no-store'}),
    ]);
    if(!summaryRes.ok || !flaggedRes.ok || !curveRes.ok) throw new Error('not found');
    const summary = await summaryRes.json();
    const topFlagged = await flaggedRes.json();
    const prCurve = await curveRes.json();
    return { summary, topFlagged, prCurve, source: 'live' };
  }catch(e){
    return { ...demoModelResult(modelId), source: 'demo' };
  }
}

function demoModelResult(modelId){
  const p = DEMO_PROFILES[modelId];
  const testN = 4000;
  const alertCount = Math.round(testN * 0.03);

  const summary = {
    model_name: modelId, dataset_size: 'Small', run_tag: '',
    metrics: {
      test: { auprc: p.auprc, auroc: p.auroc, n_positive: p.n_positive },
      test_best_threshold: { f1: p.f1, precision: p.precision, recall: p.recall },
    },
    best_threshold: p.best_threshold,
    precision_at_k: Object.fromEntries(K_OPTIONS.map(k => [String(k), { precision: p.p_at_k[k] }])),
    performance: {
      avg_ms_per_transaction: p.ms_per_txn,
      throughput_transactions_per_sec: 1000 / p.ms_per_txn,
    },
  };

  const transactions = [];
  for(let r = 1; r <= 5000; r++){
    let bucketP = p.p_at_k[5000];
    for(const k of K_OPTIONS){ if(r <= k){ bucketP = p.p_at_k[k]; break; } }
    const actual = Math.random() < bucketP ? 1 : 0;
    const score = Math.max(0.01, 0.995 - (r / 5000) * 0.9 - Math.random() * 0.03);
    const predicted = r <= alertCount ? 1 : 0;
    transactions.push({ rank: r, txn_pos: 900000 + r, score: Math.round(score*10000)/10000, predicted_label: predicted, actual_label: actual, correct: predicted === actual });
  }

  const points = [];
  for(let i = 0; i <= 100; i++){
    const thr = i / 100;
    const recall = Math.max(0.01, p.recall * Math.exp(-2.2 * Math.max(0, thr - p.best_threshold)) * (thr < p.best_threshold ? Math.min(1, 1 + (p.best_threshold - thr) * 0.6) : 1));
    const precision = Math.min(0.999, p.precision * (0.35 + 0.65 * (thr / Math.max(p.best_threshold, 0.05))));
    const f1 = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0;
    const isBest = Math.abs(thr - p.best_threshold) < 1e-9;
    points.push({ threshold: Math.round(thr*10000)/10000, precision: Math.round(Math.min(precision,0.999)*10000)/10000, recall: Math.round(Math.min(recall,1)*10000)/10000, f1: Math.round(f1*10000)/10000, alert_count: Math.max(0, Math.round(testN * recall * (p.n_positive/testN) / Math.max(precision,0.01))), is_best: isBest });
  }
  const prCurve = { n_test: testN, n_positive: p.n_positive, points };

  return { summary, topFlagged: { threshold: p.best_threshold, top_n: 5000, transactions }, prCurve };
}

async function loadAllModels(){
  document.getElementById('cmpCount').textContent = 'loading…';
  const ids = MODEL_CATALOG.map(m => m.id);
  const results = await Promise.all(ids.map(id => fetchModelResult(id)));
  cache = {};
  ids.forEach((id, i) => { cache[id] = results[i]; });

  if(!primaryModelId || !cache[primaryModelId]){
    let bestId = ids[0], bestAuprc = -1;
    ids.forEach(id => { const a = cache[id].summary.metrics.test.auprc; if(a > bestAuprc){ bestAuprc = a; bestId = id; } });
    primaryModelId = bestId;
  }
  sliderIndex = null; 
  renderAll();
}

// Model switcher
function buildSwitcher(){
  const wrap = document.getElementById('modelSwitcher');
  wrap.innerHTML = MODEL_CATALOG.map(m => {
    const r = cache[m.id];
    const active = m.id === primaryModelId;
    return `<button class="switcher-btn ${active?'active':''}" data-model="${m.id}" style="color:${active?m.color:''}">
      <span class="sw" style="background:${m.color}"></span>${m.label}
    </button>`;
  }).join('');
  wrap.querySelectorAll('.switcher-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if(primaryModelId === btn.dataset.model) return;
      primaryModelId = btn.dataset.model;
      sliderIndex = null;
      renderAll();
    });
  });
  const r = cache[primaryModelId];
  document.getElementById('switcherHint').textContent = r?.source === 'demo' ? '(demo data)' : '';
}


function buildControls(){
  const chipWrap = document.getElementById('modelChecks');
  chipWrap.innerHTML = MODEL_CATALOG.map(m => `
    <div class="chip ${selectedModels.has(m.id)?'active':''}" data-model="${m.id}" style="color:${selectedModels.has(m.id)?m.color:''}">
      <span class="sw" style="background:${m.color}"></span>${m.label}
    </div>`).join('');
  chipWrap.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const id = chip.dataset.model;
      if(selectedModels.has(id)) selectedModels.delete(id); else selectedModels.add(id);
      if(selectedModels.size === 0) selectedModels.add(id);
      renderAll();
    });
  });

  const kWrap = document.getElementById('kButtons');
  kWrap.innerHTML = K_OPTIONS.map(k => `<div class="k-btn ${k===selectedK?'active':''}" data-k="${k}">Top ${k}</div>`).join('');
  kWrap.querySelectorAll('.k-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedK = parseInt(btn.dataset.k, 10);
      renderAll();
    });
  });

  document.getElementById('pAtKHeader').textContent = `Precision @ ${selectedK}`;
  document.getElementById('rAtKHeader').textContent = `Recall @ ${selectedK}`;
}

function recallAtK(result, k){
  const nPositive = result.summary.metrics.test?.n_positive;
  if(!nPositive) return undefined;
  const hits = result.topFlagged.transactions.slice(0, k).filter(t => t.actual_label === 1).length;
  return hits / nPositive;
}

function pickBestForColumn(ids, getValue, higherIsBetter){
  let bestId = null, bestVal = null;
  ids.forEach(id => {
    const v = getValue(id);
    if(v === undefined || v === null || Number.isNaN(v)) return;
    if(bestVal === null || (higherIsBetter ? v > bestVal : v < bestVal)){
      bestVal = v;
      bestId = id;
    }
  });
  return bestId;
}

function renderComparisonTable(){
  const ids = Array.from(selectedModels).filter(id => cache[id]);
  document.getElementById('cmpCount').textContent = `${ids.length} model${ids.length!==1?'s':''} · ${cache[ids[0]]?.source === 'demo' ? 'demo data' : 'live results'}`;

  const getAuprc = id => cache[id].summary.metrics.test.auprc;
  const getAuroc = id => cache[id].summary.metrics.test.auroc;
  const getF1Best = id => cache[id].summary.metrics.test_best_threshold?.f1;
  const getPrecBest = id => cache[id].summary.metrics.test_best_threshold?.precision;
  const getRecallBest = id => cache[id].summary.metrics.test_best_threshold?.recall;
  const getPAtK = id => cache[id].summary.precision_at_k[String(selectedK)]?.precision;
  const getRAtK = id => recallAtK(cache[id], selectedK);
  const getMsPerTxn = id => cache[id].summary.performance?.avg_ms_per_transaction;
  const getThroughput = id => cache[id].summary.performance?.throughput_transactions_per_sec;


  const multiple = ids.length > 1;
  const winners = !multiple ? {} : {
    auprc: pickBestForColumn(ids, getAuprc, true),
    auroc: pickBestForColumn(ids, getAuroc, true),
    f1Best: pickBestForColumn(ids, getF1Best, true),
    precBest: pickBestForColumn(ids, getPrecBest, true),
    recallBest: pickBestForColumn(ids, getRecallBest, true),
    pAtK: pickBestForColumn(ids, getPAtK, true),
    rAtK: pickBestForColumn(ids, getRAtK, true),
    msPerTxn: pickBestForColumn(ids, getMsPerTxn, false),
    throughput: pickBestForColumn(ids, getThroughput, true),
  };

  const cmpBody = document.getElementById('cmpBody');
  cmpBody.innerHTML = ids.map(id => {
    const r = cache[id];
    const m = modelMeta(id);
    const s = r.summary;
    const pAtK = getPAtK(id);
    const rAtK = getRAtK(id);
    const perf = s.performance || {};
    const rowActive = id === primaryModelId ? 'active-row' : '';
    const cls = (key) => winners[key] === id ? 'best' : '';
    return `<tr class="${rowActive}" data-model="${id}">
      <td class="modelcell"><span class="sw" style="background:${m.color}"></span>${m.label}${r.source==='demo' ? ' <span class="flag">(demo)</span>' : ''}</td>
      <td class="${cls('auprc')}">${s.metrics.test.auprc?.toFixed(3) ?? '—'}</td>
      <td class="${cls('auroc')}">${s.metrics.test.auroc?.toFixed(3) ?? '—'}</td>
      <td class="${cls('f1Best')}">${s.metrics.test_best_threshold?.f1?.toFixed(3) ?? '—'}</td>
      <td class="${cls('precBest')}">${s.metrics.test_best_threshold?.precision?.toFixed(3) ?? '—'}</td>
      <td class="${cls('recallBest')}">${s.metrics.test_best_threshold?.recall?.toFixed(3) ?? '—'}</td>
      <td class="${cls('pAtK')}">${pAtK !== undefined ? pAtK.toFixed(3) : '—'}</td>
      <td class="${cls('rAtK')}">${rAtK !== undefined ? rAtK.toFixed(3) : '—'}</td>
      <td class="${cls('msPerTxn')}">${perf.avg_ms_per_transaction !== undefined ? perf.avg_ms_per_transaction.toFixed(3) : '—'}</td>
      <td class="${cls('throughput')}">${perf.throughput_transactions_per_sec !== undefined ? Math.round(perf.throughput_transactions_per_sec).toLocaleString() : '—'}</td>
    </tr>`;
  }).join('');

  cmpBody.querySelectorAll('tr').forEach(row => {
    row.addEventListener('click', () => {
      const id = row.dataset.model;
      if(id === primaryModelId) return;
      primaryModelId = id;
      sliderIndex = null;
      renderAll();
    });
  });

  // precision@K comparison chart
  const ctx = document.getElementById('compareChart');
  const labels = ids.map(id => modelMeta(id).label);
  const data = ids.map(id => cache[id].summary.precision_at_k[String(selectedK)]?.precision ?? 0);
  const colors = ids.map(id => modelMeta(id).color);

  try{
    if(compareChart) compareChart.destroy();
    compareChart = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ label: `Precision @ ${selectedK}`, data, backgroundColor: colors, borderRadius: 6, maxBarThickness: 46 }] },
      options: {
        responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{ display:false } },
        scales:{
          x:{ ticks:{ color:'#7c8798', font:{size:10} }, grid:{ display:false } },
          y:{ min:0, max:1, ticks:{ color:'#7c8798', font:{size:10} }, grid:{ color:'#1f2630' } },
        }
      }
    });
  }catch(e){
    console.error('compareChart failed - continuing without it', e);
    compareChart = null;
  }
}

function renderTopFlagged(){
  const r = cache[primaryModelId];
  if(!r) return;
  const m = modelMeta(primaryModelId);
  document.getElementById('primaryModelNote').innerHTML = `Ledger showing <b style="color:${m.color}">${m.label}</b> — switch models above to inspect a different one.`;
  const rows = r.topFlagged.transactions.slice(0, selectedK);
  const maxRender = Math.min(rows.length, 300);
  document.getElementById('flaggedSubtitle').textContent = `${m.label} — showing ${maxRender} of top ${rows.length}`;
  document.getElementById('flaggedBody').innerHTML = rows.slice(0, maxRender).map(t => `
    <tr>
      <td>#${t.rank}</td>
      <td>txn_${t.txn_pos}</td>
      <td class="score">${t.score.toFixed(4)}</td>
      <td><span class="badge ${t.predicted_label ? 'alert':'processed'}">${t.predicted_label ? 'FLAGGED':'CLEARED'}</span></td>
      <td>${t.actual_label ? 'FRAUD' : 'legit'}</td>
      <td><span class="badge ${t.correct?'correct':'incorrect'}">${t.correct ? '✓ correct':'✗ incorrect'}</span></td>
    </tr>`).join('');
}

// Threshold Explorer
function closestIndexForThreshold(points, thr){
  let best = 0, bestDiff = Infinity;
  points.forEach((p, i) => { const d = Math.abs(p.threshold - thr); if(d < bestDiff){ bestDiff = d; best = i; } });
  return best;
}

function bestIndex(points, fallbackThreshold){
  const exact = points.findIndex(p => p.is_best);
  return exact !== -1 ? exact : closestIndexForThreshold(points, fallbackThreshold ?? 0.5);
}

function renderThresholdReadout(points, idx){
  const p = points[idx];
  document.getElementById('roThreshold').textContent = p.threshold.toFixed(3);
  document.getElementById('roPrecision').textContent = p.precision.toFixed(3);
  document.getElementById('roRecall').textContent = p.recall.toFixed(3);
  document.getElementById('roF1').textContent = p.f1.toFixed(3);
  document.getElementById('roAlerts').textContent = p.alert_count.toLocaleString();
}

function updateThresholdMarker(idx){
  const points = cache[primaryModelId].prCurve.points;
  const p = points[idx];
  renderThresholdReadout(points, idx);
  if(!thresholdChart) return;
  try{
    thresholdChart.data.datasets[2].data = [{x: p.threshold, y: p.precision}];
    thresholdChart.data.datasets[3].data = [{x: p.threshold, y: p.recall}];
    thresholdChart.update('none');
  }catch(e){
    console.error('thresholdChart update failed', e);
    thresholdChart = null;
  }
}

function buildThresholdExplorer(){
  const r = cache[primaryModelId];
  if(!r) return;
  const m = modelMeta(primaryModelId);
  document.getElementById('explorerModelLabel').innerHTML = `<span class="sw" style="background:${m.color}; display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;"></span>${m.label}${r.source==='demo' ? ' (demo)' : ''}`;

  const points = r.prCurve.points;
  const bestIdx = bestIndex(points, r.summary.best_threshold);
  if(sliderIndex === null){
    sliderIndex = bestIdx;
  }


  const thresholds = points.map(p => p.threshold);
  const minThr = thresholds[0], maxThr = thresholds[thresholds.length - 1];
  const slider = document.getElementById('thresholdSlider');
  slider.min = minThr;
  slider.max = maxThr;
  slider.step = Math.max((maxThr - minThr) / 500, 0.0001);
  slider.value = points[sliderIndex].threshold;
  slider.oninput = () => {
    sliderIndex = closestIndexForThreshold(points, parseFloat(slider.value));
    updateThresholdMarker(sliderIndex);
  };

  const resetBtn = document.getElementById('resetToBestBtn');
  resetBtn.textContent = `Reset to best F1 (${points[bestIdx].threshold.toFixed(3)})`;
  resetBtn.onclick = () => {
    sliderIndex = bestIdx;
    slider.value = points[bestIdx].threshold;
    updateThresholdMarker(sliderIndex);
  };

  const ctx = document.getElementById('thresholdChart');
  const precisionData = points.map(p => ({x: p.threshold, y: p.precision}));
  const recallData = points.map(p => ({x: p.threshold, y: p.recall}));
  const cur = points[sliderIndex];
  const best = points[bestIdx];

  try{
    if(thresholdChart) thresholdChart.destroy();
    thresholdChart = new Chart(ctx, {
      type: 'line',
      data: { datasets: [
        { label: 'Precision', data: precisionData, borderColor: '#3ddc97', backgroundColor:'transparent', pointRadius:0, borderWidth:2, tension:.15 },
        { label: 'Recall', data: recallData, borderColor: '#5aa9ff', backgroundColor:'transparent', pointRadius:0, borderWidth:2, tension:.15 },
        { label: 'Current (precision)', data: [{x:cur.threshold, y:cur.precision}], borderColor:'#f2b84b', backgroundColor:'#f2b84b', pointRadius:6, pointHoverRadius:6, showLine:false },
        { label: 'Current (recall)', data: [{x:cur.threshold, y:cur.recall}], borderColor:'#f2b84b', backgroundColor:'#f2b84b', pointRadius:6, pointHoverRadius:6, showLine:false },
        { label: 'Best F1 threshold', data: [{x:best.threshold, y:0}, {x:best.threshold, y:1}], borderColor:'#ff5c5c', borderWidth:1.5, borderDash:[5,4], pointRadius:0, showLine:true },
      ]},
      options: {
        responsive:true, maintainAspectRatio:false,
        interaction:{mode:'nearest', intersect:false},
        plugins:{ legend:{ labels:{ color:'#7c8798', font:{family:'Inter', size:11}, filter: (item) => !item.text.startsWith('Current') } } },
      scales:{
        x:{ type:'linear', min:minThr, max:maxThr, title:{display:true,text:'threshold',color:'#4b5563',font:{size:10}}, ticks:{ color:'#4b5563', font:{size:10} }, grid:{ color:'#1f2630' } },
        y:{ min:0, max:1, ticks:{ color:'#7c8798', font:{size:10} }, grid:{ color:'#1f2630' } },
      }
    }
    });
  }catch(e){
    console.error('thresholdChart failed to initialize', e);
    thresholdChart = null;
  }

  renderThresholdReadout(points, sliderIndex);
}


function renderAll(){
  buildSwitcher();
  buildControls();
  renderComparisonTable();
  renderTopFlagged();
  buildThresholdExplorer();
}

function initAnalystTab(){
  document.getElementById('refreshBtn').addEventListener('click', loadAllModels);
  loadAllModels();
}

/* ════════════════════════════════════════════════════════════════
   LinearLearn — Python/Flask Edition
   Frontend JS: orchestrates all API calls to Flask backend
   ════════════════════════════════════════════════════════════════ */

'use strict';

// ── Global State ─────────────────────────────────────────────────────────────
const STATE = {
  sessionId:    null,
  columns:      [],
  numericCols:  [],
  catCols:      [],
  preprocessed: false,
  trained:      false,
  featureCols:  [],
  targetCol:    null,
};

// ── Utility ───────────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const qs = (sel) => document.querySelector(sel);

function setStatus(elId, type, msg) {
  const el = $(elId);
  el.className = `status-msg ${type}`;
  el.innerHTML = type === 'loading'
    ? `<span class="spinner"></span>${msg}`
    : msg;
  el.classList.remove('hidden');
}
function hideStatus(elId) { $(elId).classList.add('hidden'); }

function goToStep(n) {
  document.querySelectorAll('.step-panel').forEach(p => p.classList.remove('active'));
  $(`step${n}`).classList.add('active');
  document.querySelectorAll('.step-btn').forEach(b => {
    const s = parseInt(b.dataset.step);
    b.classList.toggle('active', s === n);
    b.classList.toggle('done',   s < n);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function buildTable(records, maxRows = 200) {
  if (!records || records.length === 0) return '<p style="color:var(--text-lo)">No data</p>';
  const cols = Object.keys(records[0]);
  const rows = records.slice(0, maxRows);
  let html = '<div class="data-table-wrap"><table><thead><tr>'
    + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr>' + cols.map(c => `<td>${r[c] ?? ''}</td>`).join('') + '</tr>').join('')
    + '</tbody></table></div>';
  if (records.length > maxRows) html += `<p style="color:var(--text-lo);font-size:12px;margin-top:6px">Showing ${maxRows} of ${records.length} rows</p>`;
  return html;
}

function buildMetricsGrid(metrics) {
  const defs = [
    { key: 'r2',     label: 'R² Score',   cls: 'good',    fmt: v => v.toFixed(4) },
    { key: 'adj_r2', label: 'Adj. R²',    cls: 'teal',    fmt: v => v.toFixed(4) },
    { key: 'rmse',   label: 'RMSE',       cls: 'warning', fmt: v => v.toFixed(2) },
    { key: 'mae',    label: 'MAE',        cls: 'alt',     fmt: v => v.toFixed(2) },
    { key: 'mse',    label: 'MSE',        cls: 'info',    fmt: v => v.toFixed(2) },
  ];
  return defs.map(d => metrics[d.key] !== undefined ? `
    <div class="metric-card">
      <div class="metric-name">${d.label}</div>
      <div class="metric-value ${d.cls}">${d.fmt(metrics[d.key])}</div>
    </div>` : '').join('');
}

function buildPlotsGrid(plots) {
  return plots.map(p => `
    <div class="plot-card">
      <img src="${p.url}" alt="${p.title}" loading="lazy" />
      <div class="plot-title">${p.title}</div>
    </div>`).join('');
}

function showToast(message, type = "success") {

    const toast = document.getElementById("toast");

    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}

// ── Step 1: Upload ────────────────────────────────────────────────────────────
const uploadZone   = $('uploadZone');
const csvFileInput = $('csvFileInput');

uploadZone.addEventListener('click', () => csvFileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleUpload(f);
});
csvFileInput.addEventListener('change', () => {
  if (csvFileInput.files[0]) handleUpload(csvFileInput.files[0]);
});
$('loadSampleBtn').addEventListener('click', () => handleUpload(null));

async function handleUpload(file) {
  showToast("⏳ Uploading dataset...", "loading");
  setStatus('uploadStatus', 'loading', 'Uploading and parsing data with pandas…');
  $('toStep2Btn').disabled = true;

  const formData = new FormData();
  if (file) formData.append('file', file);
  if (STATE.sessionId) formData.append('session_id', STATE.sessionId);

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) { setStatus('uploadStatus', 'error', '❌ ' + data.error); return; }

    STATE.sessionId   = data.session_id;
    STATE.columns     = data.columns;
    STATE.numericCols = data.numeric_cols;
    STATE.catCols     = data.categorical_cols;

    setStatus('uploadStatus', 'success',
      `✅ Dataset loaded: <strong>${data.shape[0]} rows × ${data.shape[1]} columns</strong>`);
    
    showToast(
      `✅ Dataset uploaded successfully (${data.shape[0]} rows × ${data.shape[1]} columns)`,
      "success"
      );
    
    $('toStep2Btn').disabled = false;
    goToStep(2);
    populatePreview(data);
  } catch (e) {
    showToast("❌ Dataset upload failed!", "error");
    setStatus('uploadStatus', 'error', '❌ Network error: ' + e.message);
  }
}

$('toStep2Btn').addEventListener('click', () => goToStep(2));

// ── Step 2: Preview ───────────────────────────────────────────────────────────
function populatePreview(data) {
  const missingTotal = Object.values(data.missing).reduce((a, b) => a + b, 0);
  let html = `
    <div class="info-cards">
      <div class="info-card">
        <div class="info-card-label">Rows</div>
        <div class="info-card-value">${data.shape[0]}</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Columns</div>
        <div class="info-card-value">${data.shape[1]}</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Numeric</div>
        <div class="info-card-value">${data.numeric_cols.length}</div>
        <div class="info-card-sub">${data.numeric_cols.join(', ')}</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Categorical</div>
        <div class="info-card-value">${data.categorical_cols.length}</div>
        <div class="info-card-sub">${data.categorical_cols.join(', ') || '—'}</div>
      </div>
      <div class="info-card">
        <div class="info-card-label">Missing Values</div>
        <div class="info-card-value" style="color:${missingTotal > 0 ? 'var(--warning)' : 'var(--success)'}">${missingTotal}</div>
      </div>
    </div>
    <h4 style="margin-bottom:10px;color:var(--text-mid);font-size:13px;font-family:var(--font-mono)">FIRST 10 ROWS</h4>
    ${buildTable(data.preview)}
  `;

  if (Object.keys(data.stats).length) {
    html += `<h4 style="margin:20px 0 10px;color:var(--text-mid);font-size:13px;font-family:var(--font-mono)">DESCRIPTIVE STATISTICS</h4>`;
    const statRecords = Object.entries(data.stats).map(([col, s]) => ({ column: col, ...s }));
    html += buildTable(statRecords);
  }

  $('previewContent').innerHTML = html;
  $('toStep3Btn').disabled = false;

  // Populate preprocess btn
  $('runPreprocessBtn').disabled = false;
  $('preprocessContent').innerHTML = '<p style="color:var(--text-mid);font-size:13px">Click <strong>Run Preprocessing</strong> to clean and encode the data.</p>';
}

$('toStep3Btn').addEventListener('click', () => goToStep(3));

// ── Step 3: Preprocess ────────────────────────────────────────────────────────
$('runPreprocessBtn').addEventListener('click', runPreprocess);

async function runPreprocess() {
  if (!STATE.sessionId) return;
  setStatus('preprocessStatus', 'loading', 'Running pandas preprocessing + sklearn LabelEncoder…');
  $('runPreprocessBtn').disabled = true;

  try {
    const res  = await fetch('/api/preprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: STATE.sessionId }),
    });
    const data = await res.json();

    if (data.error) {
      setStatus('preprocessStatus', 'error', '❌ ' + data.error);
      $('runPreprocessBtn').disabled = false;
      return;
    }

    STATE.preprocessed = true;
    setStatus('preprocessStatus', 'success',
      `✅ Preprocessing complete — ${data.missing_after} missing values remaining`);

    let html = `
      <div class="info-cards">
        <div class="info-card"><div class="info-card-label">Final Shape</div><div class="info-card-value">${data.shape[0]}×${data.shape[1]}</div></div>
        <div class="info-card"><div class="info-card-label">Missing After</div><div class="info-card-value" style="color:var(--success)">${data.missing_after}</div></div>
        <div class="info-card"><div class="info-card-label">Encoded Cols</div><div class="info-card-value">${Object.keys(data.encoders).length}</div></div>
      </div>`;

    if (Object.keys(data.encoders).length) {
      html += `<h4 style="margin:16px 0 10px;color:var(--text-mid);font-size:13px;font-family:var(--font-mono)">LABEL ENCODINGS</h4>
        <div class="preprocess-result-grid">` +
        Object.entries(data.encoders).map(([col, enc]) => `
          <div class="encoder-chip">
            <div class="encoder-col">${col}</div>
            <div class="encoder-map">${Object.entries(enc.mapping).map(([k, v]) => `${k} → ${v}`).join('<br/>')}</div>
          </div>`).join('') + `</div>`;
    }

    html += `<h4 style="margin:20px 0 10px;color:var(--text-mid);font-size:13px;font-family:var(--font-mono)">PROCESSED PREVIEW</h4>` + buildTable(data.preview);
    $('preprocessResult').innerHTML = html;
    $('preprocessResult').classList.remove('hidden');
    $('toStep4Btn').disabled = false;

    // Populate EDA / Config dropdowns
    populateDropdowns(data.columns);
  } catch (e) {
    setStatus('preprocessStatus', 'error', '❌ ' + e.message);
    $('runPreprocessBtn').disabled = false;
  }
}

$('toStep4Btn').addEventListener('click', () => goToStep(4));

function populateDropdowns(cols) {
  // EDA target select
  const edaSel = $('edaTargetSelect');
  edaSel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join('');
  $('edaControls').classList.remove('hidden');
  $('edaContent').innerHTML = '<p style="color:var(--text-mid);font-size:13px">Select a target column and click <strong>Generate EDA Plots</strong>.</p>';

  // Config dropdowns
  $('configContent').innerHTML = '';
  $('configForm').classList.remove('hidden');
  const targetSel  = $('targetColSelect');
  const featureSel = $('featureColSelect');
  targetSel.innerHTML  = cols.map(c => `<option value="${c}">${c}</option>`).join('');
  featureSel.innerHTML = cols.map(c => `<option value="${c}" selected>${c}</option>`).join('');
  // Default: last column as target, rest as features
  if (cols.length > 1) {
    targetSel.value = cols[cols.length - 1];
    [...featureSel.options].forEach((o, i) => { o.selected = i < cols.length - 1; });
  }
  $('toStep5Btn').disabled = false;
  $('toStep6Btn') && ($('toStep6Btn').disabled = false);
}

$('toStep5Btn') && $('toStep5Btn').addEventListener('click', () => goToStep(5));

// ── Step 4: EDA ───────────────────────────────────────────────────────────────
$('runEDABtn').addEventListener('click', runEDA);

async function runEDA() {
  if (!STATE.sessionId) return;
  const targetCol = $('edaTargetSelect').value;
  $('runEDABtn').disabled = true;
  $('runEDABtn').innerHTML = '<span class="spinner"></span>Generating Matplotlib plots…';
  $('edaPlots').innerHTML = '';

  try {
    const res  = await fetch('/api/eda', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: STATE.sessionId, target_col: targetCol }),
    });
    const data = await res.json();

    if (data.error) { alert('EDA error: ' + data.error); return; }

    $('edaPlots').innerHTML = buildPlotsGrid(data.plots);

    if (data.corr_with_target && Object.keys(data.corr_with_target).length) {
      let corrHtml = `<h4 style="margin:20px 0 10px;color:var(--text-mid);font-size:13px;font-family:var(--font-mono)">CORRELATION WITH TARGET: ${targetCol}</h4>
        <div class="coeff-section"><div class="coeff-section">` +
        Object.entries(data.corr_with_target)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .map(([k, v]) => `<div class="coeff-row"><span class="coeff-name">${k}</span><span class="coeff-val">${v}</span></div>`)
          .join('') + `</div></div>`;
      $('edaPlots').insertAdjacentHTML('beforeend', corrHtml);
    }

    $('toStep5Btn').disabled = false;
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    $('runEDABtn').disabled = false;
    $('runEDABtn').textContent = '📊 Generate EDA Plots';
  }
}

// ── Step 5: Theory tabs ───────────────────────────────────────────────────────
document.querySelectorAll('.theory-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.theory-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.theory-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ── Step 6: Config ────────────────────────────────────────────────────────────
$('testSplitRange').addEventListener('input', function () {
  $('splitLabel').textContent = this.value + '%';
});

document.querySelector('.step-panel#step6 .btn-next') &&
  document.querySelector('.step-panel#step6 .btn-next').addEventListener('click', () => {
    const targetCol  = $('targetColSelect').value;
    const featureSel = $('featureColSelect');
    const featureCols = [...featureSel.selectedOptions].map(o => o.value).filter(v => v !== targetCol);

    if (!featureCols.length) { alert('Please select at least one feature column (different from target).'); return; }

    STATE.targetCol  = targetCol;
    STATE.featureCols = featureCols;

    $('runTrainBtn').disabled = false;
    goToStep(7);
  });

// ── Step 7: Train ─────────────────────────────────────────────────────────────
$('runTrainBtn').addEventListener('click', runTrain);

async function runTrain() {
  if (!STATE.sessionId) return;
  setStatus('trainStatus', 'loading', 'Training LinearRegression with scikit-learn… generating Matplotlib plots…');
  $('runTrainBtn').disabled = true;
  $('trainResults').classList.add('hidden');

  const payload = {
    session_id:    STATE.sessionId,
    feature_cols:  STATE.featureCols,
    target_col:    STATE.targetCol,
    test_size:     parseInt($('testSplitRange').value) / 100,
    scale_features: $('scaleSelect').value === 'true',
    cv_folds:      parseInt($('cvFoldsSelect').value),
  };

  try {
    const res  = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) {
      setStatus('trainStatus', 'error', '❌ ' + data.error);
      $('runTrainBtn').disabled = false;
      return;
    }

    STATE.trained = true;
    setStatus('trainStatus', 'success',
      `✅ Training complete — ${data.n_train} train samples, ${data.n_test} test samples`);

    // Metrics
    $('metricsGrid').innerHTML = buildMetricsGrid(data.metrics);

    // Coefficients
    let coeffHtml = `<div class="coeff-section"><h4>Model Coefficients</h4>
      <div class="coeff-row"><span class="coeff-name">Intercept (β₀)</span><span class="coeff-val">${data.intercept}</span></div>` +
      Object.entries(data.coefficients).map(([k, v]) =>
        `<div class="coeff-row"><span class="coeff-name">${k}</span><span class="coeff-val">${v}</span></div>`
      ).join('') + `</div>`;
    $('coeffSection').innerHTML = coeffHtml;

    // CV
    $('cvSection').innerHTML = `<div class="cv-section"><h4>Cross-Validation (${data.cv_scores.length}-Fold) R² Scores</h4>
      <div class="coeff-row"><span class="coeff-name">Scores</span><span class="coeff-val">[${data.cv_scores.join(', ')}]</span></div>
      <div class="coeff-row"><span class="coeff-name">Mean ± Std</span><span class="coeff-val">${data.cv_mean} ± ${data.cv_std}</span></div>
    </div>`;

    // Plots
    $('trainPlots').innerHTML = buildPlotsGrid(data.plots);
    $('trainResults').classList.remove('hidden');
    $('toStep8Btn').disabled = false;

    // Build predict form
    buildPredictForm(data.coefficients);
  } catch (e) {
    setStatus('trainStatus', 'error', '❌ ' + e.message);
    $('runTrainBtn').disabled = false;
  }
}

$('toStep8Btn').addEventListener('click', () => goToStep(8));

// ── Step 8: Predict ───────────────────────────────────────────────────────────
function buildPredictForm(coefficients) {
  $('predictContent').innerHTML = '';
  $('predictForm').classList.remove('hidden');
  $('predictInputs').innerHTML = STATE.featureCols.map(f => `
    <div class="predict-input-group">
      <label>${f}</label>
      <input type="number" step="any" id="pi_${f}" placeholder="Enter ${f}" />
    </div>`).join('');
  $('runEvalBtn').disabled = false;
}

$('runPredictBtn').addEventListener('click', runPredict);

async function runPredict() {
  const inputValues = {};
  for (const f of STATE.featureCols) {
    const val = $('pi_' + f).value;
    if (val === '' || isNaN(parseFloat(val))) {
      alert(`Please enter a valid number for "${f}"`);
      return;
    }
    inputValues[f] = parseFloat(val);
  }

  $('runPredictBtn').disabled = true;
  $('runPredictBtn').innerHTML = '<span class="spinner"></span>Computing…';

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: STATE.sessionId, input_values: inputValues }),
    });
    const data = await res.json();

    if (data.error) { alert('Prediction error: ' + data.error); return; }

    const stepsHtml = data.steps.map(s =>
      `<div class="predict-step"><span class="predict-step-name">${s.step}</span><span class="predict-step-val">${s.value}</span></div>`
    ).join('');

    $('predictResult').innerHTML = `
      <div class="metric-name" style="margin-bottom:6px">Predicted ${STATE.targetCol}</div>
      <div class="predict-main">${data.prediction.toLocaleString()}</div>
      <div class="predict-steps">
        <div style="font-size:12px;color:var(--text-lo);margin-bottom:8px;font-family:var(--font-mono)">COMPUTATION BREAKDOWN</div>
        ${stepsHtml}
      </div>`;
    $('predictResult').classList.remove('hidden');
    $('toStep9Btn').disabled = false;
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    $('runPredictBtn').disabled = false;
    $('runPredictBtn').textContent = '🎯 Predict';
  }
}

$('toStep9Btn').addEventListener('click', () => goToStep(9));

// ── Step 9: Evaluate ──────────────────────────────────────────────────────────
$('runEvalBtn').addEventListener('click', runEvaluate);

async function runEvaluate() {
  $('runEvalBtn').disabled = true;
  $('runEvalBtn').innerHTML = '<span class="spinner"></span>Generating report…';

  try {
    const res  = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: STATE.sessionId }),
    });
    const data = await res.json();

    if (data.error) { alert('Evaluation error: ' + data.error); return; }

    $('evalMetricsGrid').innerHTML = buildMetricsGrid(data.metrics);
    $('evalPlots').innerHTML = buildPlotsGrid([{ title: 'Evaluation Summary', url: data.eval_plot }]);

    const rows = data.comparison;
    $('comparisonTable').innerHTML = `
      <h4 style="margin-bottom:10px;color:var(--text-mid);font-size:13px;font-family:var(--font-mono)">PREDICTED vs ACTUAL (test set, first 20 rows)</h4>` +
      buildTable(rows);

    $('evalResults').classList.remove('hidden');
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    $('runEvalBtn').disabled = false;
    $('runEvalBtn').textContent = '📈 Generate Evaluation Report';
  }
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetApp() {
  Object.assign(STATE, {
    sessionId: null, columns: [], numericCols: [], catCols: [],
    preprocessed: false, trained: false, featureCols: [], targetCol: null,
  });

  $('previewContent').innerHTML   = '<div class="placeholder-msg">👁️ Upload a dataset to see the preview</div>';
  $('preprocessContent').innerHTML = '<div class="placeholder-msg">🔧 Load data first</div>';
  $('preprocessResult').classList.add('hidden');
  $('edaContent').innerHTML       = '<div class="placeholder-msg">📊 Load and preprocess data first</div>';
  $('edaControls').classList.add('hidden');
  $('edaPlots').innerHTML         = '';
  $('configContent').innerHTML    = '<div class="placeholder-msg">⚙️ Load and preprocess data first</div>';
  $('configForm').classList.add('hidden');
  $('trainResults').classList.add('hidden');
  $('predictForm').classList.add('hidden');
  $('predictContent').innerHTML   = '<div class="placeholder-msg">🎯 Train the model first</div>';
  $('evalResults').classList.add('hidden');

  hideStatus('uploadStatus');
  hideStatus('preprocessStatus');
  hideStatus('trainStatus');

  ['toStep2Btn','toStep3Btn','toStep4Btn','toStep5Btn','toStep8Btn','toStep9Btn','runTrainBtn','runEvalBtn','runPreprocessBtn'].forEach(id => {
    const el = $(id);
    if (el) el.disabled = true;
  });

  goToStep(1);
}

// ── Step 6 next button wiring (inline with HTML button onclick) ───────────────
document.addEventListener('DOMContentLoaded', () => {
  const step6Next = document.querySelector('#step6 .btn-next');
  if (step6Next) {
    step6Next.id = 'toStep7Btn';
    step6Next.addEventListener('click', () => {
      const targetCol  = $('targetColSelect').value;
      const featureSel = $('featureColSelect');
      const featureCols = [...featureSel.selectedOptions].map(o => o.value).filter(v => v !== targetCol);

      if (!featureCols.length) {
        alert('Please select at least one feature column (different from target).');
        return;
      }

      STATE.targetCol   = targetCol;
      STATE.featureCols = featureCols;
      $('runTrainBtn').disabled = false;
      goToStep(7);
    });
  }
});
function validateStep6() {
    const targetCol = $('targetColSelect').value;

    const featureCols = [...$('featureColSelect').selectedOptions]
        .map(o => o.value)
        .filter(v => v !== targetCol);

    $('toStep7Btn').disabled = !(targetCol && featureCols.length > 0);
}

$('targetColSelect').addEventListener('change', validateStep6);
$('featureColSelect').addEventListener('change', validateStep6);
// Vanilla JS front-end for the hallucination pipeline demo.
// Talks to /api/* endpoints on the same Flask process.

const $ = (id) => document.getElementById(id);

// Metrics whose `mean_*` keys come from compute_aggregate() in
// src/evaluation/report.py. Order is the order they render in the UI.
const EVAL_BASE_METRICS = [
  "claim_precision", "claim_recall", "claim_f1",
  "retrieval_recall_at_5", "retrieval_precision_at_5",
  "retrieval_mrr", "retrieval_ndcg",
  "verification_success_rate", "end_to_end_accuracy", "confidence",
];
// Metrics only produced by scripts/evaluate_full.py — shown when present.
const EVAL_FULL_METRICS = [
  "detection_accuracy", "detection_precision", "detection_recall",
  "detection_f1", "detection_roc_auc",
  "hallucination_reduction_rate", "claim_correction_rate",
];

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data;
  try { data = await res.json(); } catch { data = {}; }
  return { ok: res.ok, status: res.status, data };
}

function setStatus(state, text) {
  const dot = $("status-dot");
  const lbl = $("status-text");
  dot.classList.remove("ok", "warn", "err");
  if (state) dot.classList.add(state);
  lbl.textContent = text;
}

function decisionClass(decision) {
  if (decision === "PASS") return "pass";
  if (decision === "REGEN") return "regen";
  return "abstain";
}

function escapeHTML(s) {
  return (s ?? "").toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderResult(d) {
  const conf = (d.confidence ?? 0).toFixed(2);
  const decision = d.decision ?? "UNKNOWN";
  const cls = decisionClass(decision);
  const claims = d.claims || [];
  const evidence = d.evidence || [];
  const traces = d.step_traces || [];

  const summary = `
    <div class="summary">
      <span class="tag ${cls}">${escapeHTML(decision)}</span>
      <span class="confidence">confidence ${conf}</span>
      <span class="bar"><span style="width:${Math.round(Number(conf) * 100)}%"></span></span>
      <span class="tag">claims: ${claims.length}</span>
      <span class="tag">evidence: ${evidence.length}</span>
    </div>
    <h3 class="muted small">Final answer</h3>
    <div style="background:var(--panel-2); padding:12px; border-radius:8px; border:1px solid var(--border); margin-bottom:14px;">${escapeHTML(d.answer || "")}</div>
  `;

  const claimsHTML = claims.map((c, i) => {
    const v = (c.verdict || "unknown").toLowerCase();
    return `
      <div class="claim ${v}">
        <span class="verdict">${escapeHTML(c.verdict || "unknown")}</span>
        <span class="text">${escapeHTML(c.text)}</span>
        <div class="meta">confidence=${(c.confidence ?? 0).toFixed(2)} · provenance=${escapeHTML(c.provenance)} · citation=${escapeHTML(c.citation || "-")}</div>
      </div>
    `;
  }).join("");

  const evidenceHTML = evidence.map((e) => `
    <li><b>${escapeHTML(e.doc_id)}</b> <span class="muted small">score=${(e.score ?? 0).toFixed(3)}</span><br>${escapeHTML(e.text)}</li>
  `).join("");

  const tracesHTML = traces.map((t) => `
    <details>
      <summary>Step ${t.step} — ${escapeHTML(t.name)} — ${(t.duration_ms ?? 0).toFixed(1)} ms</summary>
      <pre>${escapeHTML(JSON.stringify(t.payload ?? {}, null, 2))}</pre>
    </details>
  `).join("");

  $("result").innerHTML = `
    ${summary}
    <div class="claims"><h3>Claims (per-claim verdicts)</h3>${claimsHTML || "<p class='muted'>No claims extracted.</p>"}</div>
    <div class="evidence"><h3>Evidence (top retrieved)</h3><ul>${evidenceHTML || "<li class='muted'>No evidence.</li>"}</ul></div>
    <div class="traces">
      <h3>Step traces (13 steps)</h3>
      ${tracesHTML}
    </div>
    <p class="muted small" id="result-mode-note">Pipeline ran.</p>
  `;
}

async function ping() {
  const r = await fetchJSON("/api/health");
  if (!r.ok) { setStatus("err", "server unreachable"); return; }
  const { index_loaded, n_documents } = r.data;
  if (!index_loaded) setStatus("warn", "empty index — build one");
  else setStatus("ok", `${n_documents} chunks indexed`);
  $("index-meta").textContent = index_loaded ? `current: ${n_documents} chunks` : "no index yet";
  updateModeBadge();
}

async function updateModeBadge() {
  const badge = $("mode-badge");
  if (!badge) return;
  const r = await fetchJSON("/models/status");
  if (!r.ok) { badge.textContent = "?"; return; }
  const mode = (r.data && r.data.mode) || "unknown";
  badge.classList.remove("live", "degraded");
  if (mode === "live") badge.classList.add("live");
  else badge.classList.add("degraded");
  badge.textContent = mode === "live" ? "live mode" : "degraded";
}

function renderStatusRow(c) {
  const active = !!c.active;
  const pill = active
    ? `<span class="pill pill-active">● ACTIVE</span>`
    : `<span class="pill pill-inactive">○ INACTIVE</span>`;
  return `
    <tr>
      <td><b>${escapeHTML(c.name)}</b><div class="muted small">${escapeHTML(c.category)}</div></td>
      <td><code>${escapeHTML(c.backend)}</code></td>
      <td><code class="model-name">${escapeHTML(c.configured_model || "unknown")}</code></td>
      <td>${pill}</td>
      <td class="muted small">${escapeHTML(c.reason || "")}</td>
    </tr>
  `;
}

async function loadModelStatus() {
  const tbody = $("status-tbody");
  const modeEl = $("status-mode");
  const summaryEl = $("status-summary");
  if (!tbody) return;
  const r = await fetchJSON("/models/status");
  if (!r.ok) {
    tbody.innerHTML = `<tr><td colspan="5" class="error">Failed to load model status.</td></tr>`;
    if (modeEl) modeEl.textContent = "unreachable";
    return;
  }
  const data = r.data || {};
  const components = Array.isArray(data.components) ? data.components : [];
  if (!components.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="muted small">No components reported.</td></tr>`;
  } else {
    tbody.innerHTML = components.map(renderStatusRow).join("");
  }
  if (modeEl) modeEl.textContent = `mode: ${escapeHTML(data.mode || "unknown")}`;
  if (summaryEl) summaryEl.textContent = escapeHTML(data.summary || "");
}

// ---- Dataset selector (shared by Accuracy + Evaluation) ---------------

const DATASET_KEY = "halluc_active_dataset";

let _datasetState = {
  list: [],                  // [{slug, label, available, n_queries}]
  active: "",                // currently selected slug
};

function loadActiveDataset() {
  try { return localStorage.getItem(DATASET_KEY) || ""; }
  catch { return ""; }
}
function saveActiveDataset(slug) {
  try { localStorage.setItem(DATASET_KEY, slug); }
  catch { /* ignore */ }
}

function renderDatasetPills(targetId) {
  const host = $(targetId);
  if (!host) return;
  host.innerHTML = _datasetState.list.map((d) => {
    const cls = d.slug === _datasetState.active ? " active" : "";
    const available = d.available ? "true" : "false";
    const count = d.n_queries ? `${d.n_queries}` : "—";
    return `
      <button class="dataset-pill${cls}" data-slug="${escapeHTML(d.slug)}" data-available="${available}">
        ${escapeHTML(d.label)}
        <span class="pill-count">${count}</span>
      </button>
    `;
  }).join("");
  host.querySelectorAll(".dataset-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      const slug = btn.dataset.slug;
      const ds = _datasetState.list.find((x) => x.slug === slug);
      if (!ds || !ds.available) return;
      _datasetState.active = slug;
      saveActiveDataset(slug);
      renderDatasetPills("acc-dataset-pills");
      renderDatasetPills("eval-dataset-pills");
      loadAccuracy();
      loadEvaluation();
    });
  });
}

async function loadDatasetList() {
  const r = await fetchJSON("/api/evaluation/datasets");
  if (!r.ok) return;
  const data = r.data || {};
  const list = Array.isArray(data.datasets) ? data.datasets : [];
  _datasetState.list = list.map((d) => ({
    slug: d.slug,
    label: d.label,
    available: !!d.available,
    n_queries: (d.report && d.report.n_queries) || 0,
  }));
  const saved = loadActiveDataset();
  const savedInList = _datasetState.list.find((d) => d.slug === saved && d.available);
  if (savedInList) {
    _datasetState.active = saved;
  } else if (data.default_slug) {
    _datasetState.active = data.default_slug;
  } else if (_datasetState.list.length) {
    const first = _datasetState.list.find((d) => d.available);
    _datasetState.active = first ? first.slug : _datasetState.list[0].slug;
  }
  renderDatasetPills("acc-dataset-pills");
  renderDatasetPills("eval-dataset-pills");
}

// ---- Comparison panel -------------------------------------------------

function deltaCell(delta) {
  if (delta === null || delta === undefined) return `<td class="delta-zero">—</td>`;
  if (delta > 0)  return `<td class="delta-pos">+${fmt(delta, 3)}</td>`;
  if (delta < 0)  return `<td class="delta-neg">${fmt(delta, 3)}</td>`;
  return `<td class="delta-zero">0.000</td>`;
}

function stripBold(s) {
  return (s || "").replace(/\*\*/g, "");
}

async function loadWhyWin() {
  const r = await fetchJSON("/api/literature/headline");
  if (!r.ok) return;
  const data = r.data || {};
  const tally = data.tally || {};
  const win  = tally.win  || 0;
  const tie  = tally.tie  || 0;
  const loss = tally.loss || 0;
  const total = win + tie + loss || 1;

  const bar = $("whywin-bar");
  if (bar) {
    bar.innerHTML = `
      <div class="seg-win"  style="width:${(win/total*100).toFixed(1)}%"></div>
      <div class="seg-tie"  style="width:${(tie/total*100).toFixed(1)}%"></div>
      <div class="seg-loss" style="width:${(loss/total*100).toFixed(1)}%"></div>
    `;
  }
  const meta = $("whywin-meta");
  if (meta) meta.innerHTML = `<b>${win} wins</b> · ${tie} at par · ${loss} trails &nbsp;·&nbsp; ${total} evaluation dimensions &nbsp;·&nbsp; ${data.methods_compared?.length ?? 11} papers compared`;
  const blurb = $("whywin-blurb");
  if (blurb && tally.verdict) blurb.innerHTML = escapeHTML(stripBold(tally.verdict));

  const grid = $("whywin-grid");
  if (grid) {
    const wins = data.wins || [];
    const losses = data.losses || [];
    grid.innerHTML = `
      <div class="win-card win-card-win">
        <h3>✓ Wins <span class="muted small">${wins.length}</span></h3>
        <ul>${wins.map((v) => `
          <li>
            <div class="dim-name">${escapeHTML(stripBold(v.dimension))}</div>
            <div class="dim-best">ours: ${escapeHTML(stripBold(v.our_score))}</div>
          </li>
        `).join("")}</ul>
      </div>
      <div class="win-card win-card-loss">
        <h3>✗ Trails <span class="muted small">${losses.length}</span></h3>
        <ul>${losses.map((v) => `
          <li>
            <div class="dim-name">${escapeHTML(stripBold(v.dimension))}</div>
            <div class="dim-best">ours: ${escapeHTML(stripBold(v.our_score))}</div>
          </li>
        `).join("")}</ul>
      </div>
    `;
  }
}

async function loadComparison() {
  const head = $("comparison-head");
  const body = $("comparison-body");
  if (!head || !body) return;
  const r = await fetchJSON("/api/evaluation/comparison");
  if (!r.ok) {
    body.innerHTML = `<tr><td class="muted small">Failed to load comparison.</td></tr>`;
    return;
  }
  const data = r.data || {};
  const slugs = (data.datasets || []).map((d) => d.slug);
  if (!slugs.length) {
    body.innerHTML = `<tr><td class="muted small">No datasets registered.</td></tr>`;
    return;
  }
  // Header row: Metric + one column per dataset + Δ.
  head.innerHTML = `
    <th>Metric</th>
    ${slugs.map((s) => `<th class="col-dataset">${escapeHTML(s)}</th>`).join("")}
    <th>Δ</th>
  `;
  const rows = Array.isArray(data.rows) ? data.rows : [];
  body.innerHTML = rows.length
    ? rows.map((row) => {
        const cells = slugs.map((s) => `<td>${fmt(row[s], 3)}</td>`).join("");
        return `<tr><td>${escapeHTML(row.metric)}</td>${cells}${deltaCell(row.delta)}</tr>`;
      }).join("")
    : `<tr><td colspan="${2 + slugs.length}" class="muted small">
         No comparison metrics yet. Run <code>scripts/evaluate_datasets.py</code> to populate.
       </td></tr>`;
}

// ---- Evaluation panel ---------------------------------------------------

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function colorForScore(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "";
  if (score >= 0.7) return "metric-card-green";
  if (score >= 0.3) return "metric-card-yellow";
  return "metric-card-red";
}

function prettyMetricName(key) {
  // "claim_f1" -> "Claim F1", "retrieval_ndcg" -> "Retrieval NDCG"
  return key
    .split("_")
    .map((w) => (w === w.toUpperCase() ? w : w[0].toUpperCase() + w.slice(1)))
    .join(" ");
}

function renderEvalMetricCard(name, mean, stddev) {
  const cls = colorForScore(mean);
  const naCls = mean === null || mean === undefined ? " metric-card-na" : "";
  const stddevLine = (stddev !== null && stddev !== undefined)
    ? `<div class="metric-stddev">± ${fmt(stddev)}</div>`
    : "";
  return `
    <div class="metric-card ${cls}${naCls}">
      <div class="metric-name">${escapeHTML(name)}</div>
      <div class="metric-value">${fmt(mean)}</div>
      ${stddevLine}
    </div>
  `;
}

async function loadEvaluation() {
  const grid = $("eval-aggregate-grid");
  const rowCountEl = $("eval-row-count");
  const lastUpdEl = $("eval-last-updated");
  const sourceEl = $("eval-source");
  const dsEl = $("eval-dataset");
  const srcFileEl = $("eval-source-file");
  if (!grid) return;

  const slug = _datasetState.active || "";
  const url = slug ? `/api/evaluation?dataset=${encodeURIComponent(slug)}` : "/api/evaluation";
  const r = await fetchJSON(url);
  if (!r.ok) {
    grid.innerHTML = `<div class="empty-note">Failed to load evaluation.</div>`;
    if (sourceEl) sourceEl.textContent = "unreachable";
    return;
  }
  const data = r.data || {};
  const aggregate = data.aggregate || {};
  const rows = Array.isArray(data.per_query) ? data.per_query : [];

  // Meta line.
  if (rowCountEl) rowCountEl.textContent = `${data.n_queries ?? rows.length}`;
  if (lastUpdEl) lastUpdEl.textContent = data.last_modified ? data.last_modified.replace("T", " ").replace(/\.\d+/, "") : "never";
  if (sourceEl) sourceEl.textContent = data.source || "unknown";
  if (dsEl) dsEl.textContent = slug || "default";
  if (srcFileEl && data.source_file) {
    srcFileEl.textContent = data.source_file.replace(/^.*\/data\//, "data/");
  }

  // Aggregate grid: BASE then FULL (only when full metrics are present).
  const cards = [];
  for (const key of EVAL_BASE_METRICS) {
    const mean = aggregate[`mean_${key}`];
    const stddev = aggregate[`stdev_${key}`];
    if (mean === undefined) continue;
    cards.push(renderEvalMetricCard(prettyMetricName(key), mean, stddev));
  }
  for (const key of EVAL_FULL_METRICS) {
    if (aggregate[key] === undefined) continue;
    cards.push(renderEvalMetricCard(prettyMetricName(key), aggregate[key], null));
  }
  grid.innerHTML = cards.length
    ? cards.join("")
    : `<div class="empty-note">No metrics available. Run <code>scripts/evaluate.py</code> first.</div>`;
}

// ---- Accuracy panel ---------------------------------------------------

function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function tileClass(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "acc-tile-na";
  if (score >= 0.7) return "acc-tile-good";
  if (score >= 0.4) return "acc-tile-mid";
  return "acc-tile-bad";
}

function setAccTile(tileId, value) {
  const tile = $(tileId);
  if (!tile) return;
  tile.classList.remove("acc-tile-good", "acc-tile-mid", "acc-tile-bad", "acc-tile-na");
  tile.classList.add(tileClass(value));
  const valueEl = tile.querySelector(".acc-tile-value");
  if (valueEl) valueEl.textContent = fmtPct(value);
}

function setAccDonut(correct, total) {
  const arc = $("acc-donut-arc");
  const valueEl = $("acc-donut-value");
  if (!arc || !valueEl) return;
  const circumference = 2 * Math.PI * 52; // r=52
  const ratio = total > 0 ? correct / total : 0;
  const filled = circumference * ratio;
  arc.setAttribute("stroke-dasharray", `${filled.toFixed(2)} ${circumference.toFixed(2)}`);
  arc.setAttribute("stroke", ratio >= 0.7 ? "var(--green)" : ratio >= 0.4 ? "var(--yellow)" : "var(--red)");
  valueEl.textContent = total > 0 ? `${correct}/${total}` : "—";
}

function setAccDecisionBar(name, count, total) {
  const bar = $(`acc-bar-${name}`);
  const lbl = $(`acc-count-${name}`);
  if (bar) bar.style.width = total > 0 ? `${(count / total) * 100}%` : "0%";
  if (lbl) lbl.textContent = `${count}`;
}

function renderAccRow(row, idx) {
  const decision = (row.decision || "").toLowerCase();
  const decisionCls = ["pass", "regen", "abstain"].includes(decision) ? decision : "";
  const pill = decisionCls
    ? `<span class="tag ${decisionCls}">${escapeHTML(row.decision || "—")}</span>`
    : `<span class="muted small">—</span>`;
  const correct = row.correct
    ? `<span class="tag pass">✓ correct</span>`
    : `<span class="tag abstain">✗ wrong</span>`;
  const query = (row.query || "").toString().replace(/\s+/g, " ").trim();
  return `
    <tr>
      <td class="muted small">${idx + 1}</td>
      <td class="query-cell" title="${escapeHTML(query)}">${escapeHTML(query)}</td>
      <td>${pill}</td>
      <td>${correct}</td>
      <td>${fmt(row.end_to_end_accuracy, 3)}</td>
      <td>${fmt(row.claim_f1, 3)}</td>
      <td>${fmt(row.confidence, 3)}</td>
    </tr>
  `;
}

async function loadAccuracy() {
  const headline = $("acc-headline");
  const tbody = $("acc-rows");
  if (!headline || !tbody) return;

  const slug = _datasetState.active || "";
  const url = slug ? `/api/accuracy?dataset=${encodeURIComponent(slug)}` : "/api/accuracy";
  const r = await fetchJSON(url);
  if (!r.ok) {
    headline.innerHTML = `<div class="empty-note">Failed to load accuracy.</div>`;
    tbody.innerHTML = `<tr><td colspan="7" class="error">Failed to load accuracy.</td></tr>`;
    return;
  }

  const data = r.data || {};
  const h = data.headline || {};
  const breakdown = data.breakdown || { decisions: {}, total: 0, correct: 0, incorrect: 0 };
  const meta = data.meta || {};
  const rows = Array.isArray(data.per_query) ? data.per_query : [];

  // Meta line
  const sourceEl = $("acc-source");
  const rowCountEl = $("acc-row-count");
  const lastUpdEl = $("acc-last-updated");
  const dsEl = $("acc-dataset");
  if (sourceEl) sourceEl.textContent = `source: ${meta.source || "unknown"}`;
  if (rowCountEl) rowCountEl.textContent = meta.n_queries || rows.length || 0;
  if (lastUpdEl) {
    lastUpdEl.textContent = meta.last_modified
      ? meta.last_modified.replace("T", " ").replace(/\.\d+/, "")
      : "never";
  }
  if (dsEl) dsEl.textContent = (meta.dataset && meta.dataset !== "unknown") ? meta.dataset : (slug || "default");

  // Headline tiles
  setAccTile("acc-tile-e2e", h.end_to_end_accuracy);
  setAccTile("acc-tile-detection", h.detection_accuracy);
  setAccTile("acc-tile-claim", h.mean_claim_f1);
  setAccTile("acc-tile-verif", h.verification_success_rate);

  // Donut + counts
  setAccDonut(breakdown.correct || 0, breakdown.total || 0);
  const correctEl = $("acc-correct-count");
  const incorrectEl = $("acc-incorrect-count");
  if (correctEl) correctEl.textContent = breakdown.correct || 0;
  if (incorrectEl) incorrectEl.textContent = breakdown.incorrect || 0;

  // Decision bars
  const decs = breakdown.decisions || {};
  const total = breakdown.total || 0;
  setAccDecisionBar("pass", decs.PASS || 0, total);
  setAccDecisionBar("regen", decs.REGEN || 0, total);
  setAccDecisionBar("abstain", decs.ABSTAIN || 0, total);

  // Per-query table
  tbody.innerHTML = rows.length
    ? rows.map(renderAccRow).join("")
    : `<tr><td colspan="7" class="muted small">No rows yet — run <code>scripts/evaluate.py</code> first.</td></tr>`;
}

async function buildIndex() {
  const docs = $("docs").value.split(/\n+/).map(s => s.trim()).filter(Boolean);
  if (!docs.length) { alert("Paste at least one document."); return; }
  $("btn-index").disabled = true;
  $("btn-index").textContent = "Building…";
  const r = await fetchJSON("/api/index", { method: "POST", body: JSON.stringify({ documents: docs }) });
  $("btn-index").disabled = false;
  $("btn-index").textContent = "Build index";
  if (!r.ok) { alert(r.data.error || "Index build failed"); return; }
  await ping();
}

async function runQuery() {
  const q = $("query").value.trim();
  if (!q) { alert("Type a question."); return; }
  $("btn-query").disabled = true;
  $("btn-query").textContent = "Running…";
  $("result").innerHTML = `<p class="loading">running 13-step pipeline…</p>`;
  const r = await fetchJSON("/api/query", { method: "POST", body: JSON.stringify({ query: q }) });
  $("btn-query").disabled = false;
  $("btn-query").textContent = "Run pipeline";
  if (!r.ok) {
    $("result").innerHTML = `<div class="error">${escapeHTML(r.data.error || "query failed")}</div>`;
    return;
  }
  renderResult(r.data);
  // Reflect current mode on the result footer.
  const modeRes = await fetchJSON("/models/status");
  const note = $("result-mode-note");
  if (note && modeRes.ok) {
    const m = (modeRes.data && modeRes.data.mode) || "unknown";
    note.textContent = m === "live"
      ? "Pipeline ran with real HuggingFace models."
      : m === "degraded"
        ? `Pipeline ran in degraded mode — some components couldn't load. See Model Status above.`
        : "Pipeline ran.";
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  $("btn-index").addEventListener("click", buildIndex);
  $("btn-query").addEventListener("click", runQuery);
  const refreshBtn = $("btn-status-refresh");
  if (refreshBtn) refreshBtn.addEventListener("click", loadModelStatus);
  const evalRefreshBtn = $("btn-eval-refresh");
  if (evalRefreshBtn) evalRefreshBtn.addEventListener("click", loadEvaluation);
  const accRefreshBtn = $("btn-acc-refresh");
  if (accRefreshBtn) accRefreshBtn.addEventListener("click", loadAccuracy);
  const cmpRefreshBtn = $("btn-cmp-refresh");
  if (cmpRefreshBtn) cmpRefreshBtn.addEventListener("click", loadComparison);
  const whywinRefreshBtn = $("btn-whywin-refresh");
  if (whywinRefreshBtn) whywinRefreshBtn.addEventListener("click", loadWhyWin);
  $("query").addEventListener("keydown", (e) => { if (e.key === "Enter") runQuery(); });
  ping();
  loadModelStatus();
  setInterval(loadModelStatus, 5000);
  // Dataset list feeds both selectors + comparison view; load first.
  await loadDatasetList();
  loadAccuracy();
  setInterval(loadAccuracy, 10000);
  loadEvaluation();
  setInterval(loadEvaluation, 10000);
  loadComparison();
  setInterval(loadComparison, 15000);
  loadWhyWin();
  setInterval(loadWhyWin, 20000);
});

// Evaluation dashboard — talks to /api/evaluation and /api/accuracy.
// Reads the same data/metrics.json that scripts/evaluate*.py produces.
// Multi-dataset aware: /api/evaluation/datasets lists available datasets;
// the selector in the UI picks one and refreshes the per-dataset view,
// while /api/evaluation/comparison shows a side-by-side table.

const $ = (id) => document.getElementById(id);

const BASE_METRICS = [
  { key: "claim_precision",             label: "Claim Precision",   group: "claims" },
  { key: "claim_recall",                label: "Claim Recall",      group: "claims" },
  { key: "claim_f1",                    label: "Claim F1",          group: "claims" },
  { key: "retrieval_recall_at_5",       label: "Retrieval R@5",     group: "retrieval" },
  { key: "retrieval_precision_at_5",    label: "Retrieval P@5",     group: "retrieval" },
  { key: "retrieval_mrr",               label: "Retrieval MRR",     group: "retrieval" },
  { key: "retrieval_ndcg",              label: "Retrieval nDCG",    group: "retrieval" },
  { key: "verification_success_rate",   label: "Verification SR",   group: "verification" },
  { key: "end_to_end_accuracy",         label: "End-to-End Acc.",   group: "verification" },
  { key: "confidence",                  label: "Confidence",        group: "verification" },
];

const FULL_METRICS = [
  { key: "detection_accuracy",              label: "Detection Acc.",       group: "detection" },
  { key: "detection_precision",             label: "Detection Precision",  group: "detection" },
  { key: "detection_recall",                label: "Detection Recall",     group: "detection" },
  { key: "detection_f1",                    label: "Detection F1",         group: "detection" },
  { key: "detection_roc_auc",               label: "Detection ROC-AUC",    group: "detection" },
  { key: "hallucination_reduction_rate",    label: "Hallucination Reduction (HRR)", group: "impact" },
  { key: "claim_correction_rate",           label: "Claim Correction Rate (CCR)",    group: "impact" },
];

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmtNum(v, d = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(d);
}

function escapeHTML(s) {
  return (s ?? "").toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setDot(state, text) {
  const dot = $("status-dot");
  const lbl = $("status-text");
  dot.classList.remove("ok", "warn", "err");
  if (state) dot.classList.add(state);
  lbl.textContent = text;
}

function tileClass(value) {
  if (value == null || Number.isNaN(value)) return "acc-tile-na";
  if (value >= 0.75) return "acc-tile-good";
  if (value >= 0.5)  return "acc-tile-mid";
  return "acc-tile-bad";
}

function decisionClass(dec) {
  if (dec === "PASS") return "pass";
  if (dec === "REGEN") return "regen";
  return "abstain";
}

function pickAggregate(agg, key) {
  if (agg == null) return null;
  if (agg[key] != null) return agg[key];
  const meanKey = "mean_" + key;
  return agg[meanKey] != null ? agg[meanKey] : null;
}

function pickStddev(agg, key) {
  if (agg == null) return null;
  const stdKey = "stdev_" + key;
  return agg[stdKey] != null ? agg[stdKey] : null;
}

function setTile(id, value, label, sub) {
  const el = $(id);
  if (!el) return;
  el.classList.remove("acc-tile-good", "acc-tile-mid", "acc-tile-bad", "acc-tile-na");
  el.classList.add(tileClass(value));
  el.querySelector(".acc-tile-value").textContent = value == null ? "—" : fmtPct(value);
  if (sub) el.querySelector(".acc-tile-sub").textContent = sub;
}

function renderAggregateGrid(containerId, metrics, agg) {
  const root = $(containerId);
  root.innerHTML = "";
  let rendered = 0;
  metrics.forEach((m) => {
    const val = pickAggregate(agg, m.key);
    if (val == null) return;
    rendered += 1;
    const std = pickStddev(agg, m.key);
    const card = document.createElement("div");
    card.className = "metric-card";
    card.classList.add(
      val >= 0.75 ? "metric-card-green" :
      val >= 0.5  ? "metric-card-yellow" :
      "metric-card-red"
    );
    card.innerHTML = `
      <span class="metric-name">${escapeHTML(m.label)}</span>
      <span class="metric-value">${fmtPct(val)}</span>
      ${std != null ? `<span class="metric-stddev">± ${fmtNum(std, 3)}</span>` : ""}
    `;
    root.appendChild(card);
  });
  if (rendered === 0) {
    root.innerHTML = `<div class="empty-note">No metrics in this category.</div>`;
  }
}

function renderMetricBars(agg) {
  const root = $("eval-bars");
  root.classList.remove("muted");
  root.innerHTML = "";
  let rendered = 0;
  BASE_METRICS.forEach((m) => {
    const val = pickAggregate(agg, m.key);
    if (val == null) return;
    rendered += 1;
    const std = pickStddev(agg, m.key) || 0;
    const row = document.createElement("div");
    row.className = "eval-bar-row";
    const cls = val >= 0.75 ? "metric-card-green" : val >= 0.5 ? "metric-card-yellow" : "metric-card-red";
    row.innerHTML = `
      <span class="eval-bar-label">${escapeHTML(m.label)}</span>
      <span class="eval-bar-track">
        <span class="eval-bar-fill ${cls.replace("metric-card-","fill-")}" style="width:${Math.max(0, Math.min(1, val)) * 100}%"></span>
        <span class="eval-bar-err" style="left:calc(${Math.max(0, Math.min(1, val - std)) * 100}% ); width:${Math.max(0, Math.min(1, 2 * std)) * 100}%"></span>
      </span>
      <span class="eval-bar-value">${fmtPct(val)}</span>
    `;
    root.appendChild(row);
  });
  if (rendered === 0) {
    root.classList.add("muted");
    root.innerHTML = "No metrics available.";
  }
}

function renderTable(rows) {
  const body = $("rows-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="14" class="muted small">No rows.</td></tr>`;
    $("row-count").textContent = "0 rows";
    return;
  }
  const filter = ($("row-filter").value || "").toLowerCase().trim();
  const decFilter = $("row-decision").value;
  const filtered = rows.filter((r) => {
    if (decFilter && r.decision !== decFilter) return false;
    if (filter && !(r.query || "").toLowerCase().includes(filter)) return false;
    return true;
  });

  const html = filtered.map((r, i) => {
    const dec = r.decision || "";
    const e2e = r.end_to_end_accuracy;
    const correct = e2e ? "✓" : "✗";
    const correctClass = e2e ? "cell-good" : "cell-bad";
    return `
      <tr>
        <td>${i + 1}</td>
        <td class="query-cell">${escapeHTML(r.query || "")}</td>
        <td><span class="acc-decision-tag tag ${decisionClass(dec)}">${escapeHTML(dec)}</span></td>
        <td class="${correctClass}">${correct}</td>
        <td>${fmtPct(e2e)}</td>
        <td>${fmtPct(r.claim_f1)}</td>
        <td>${fmtPct(r.claim_precision)}</td>
        <td>${fmtPct(r.claim_recall)}</td>
        <td>${fmtPct(r.retrieval_recall_at_5)}</td>
        <td>${fmtPct(r.retrieval_precision_at_5)}</td>
        <td>${fmtPct(r.retrieval_mrr)}</td>
        <td>${fmtPct(r.retrieval_ndcg)}</td>
        <td>${fmtPct(r.verification_success_rate)}</td>
        <td>${fmtNum(r.confidence, 2)}</td>
      </tr>
    `;
  }).join("");
  body.innerHTML = html || `<tr><td colspan="14" class="muted small">No matches.</td></tr>`;
  $("row-count").textContent = `${filtered.length} / ${rows.length} rows`;
}

function renderOverview(report) {
  const agg = report.aggregate || {};
  const meta = report.meta || {};
  const rows = report.per_query || [];

  $("overview-meta").innerHTML =
    `Source: <b>${escapeHTML(meta.source || report.source || "unknown")}</b> · ` +
    `<b>${report.n_queries || 0}</b> queries · ` +
    `last updated <b>${escapeHTML(report.last_modified || "—")}</b>`;

  const e2e = pickAggregate(agg, "end_to_end_accuracy");
  const claimF1 = pickAggregate(agg, "claim_f1");
  const vsrr = pickAggregate(agg, "verification_success_rate");
  const mrr = pickAggregate(agg, "retrieval_mrr");

  setTile("tile-e2e", e2e);
  setTile("tile-claim", claimF1);
  setTile("tile-vsrr", vsrr);
  setTile("tile-mrr", mrr);

  // Source badge
  const srcBadge = $("source-badge");
  const src = report.source || "unknown";
  srcBadge.textContent = src;
  srcBadge.classList.remove("live", "degraded");
  if (src === "evaluate_full" || src === "evaluate") srcBadge.classList.add("live");
  else srcBadge.classList.add("degraded");

  // Subtitle
  $("eval-subtitle").textContent =
    `${report.n_queries || 0} queries · ${Object.keys(agg).length} aggregate metrics · source: ${src}`;
}

function renderDecisions(report) {
  const rows = report.per_query || [];
  let pass = 0, regen = 0, abstain = 0, correct = 0;
  rows.forEach((r) => {
    const d = (r.decision || "").toUpperCase();
    if (d === "PASS") pass += 1;
    else if (d === "REGEN") regen += 1;
    else if (d === "ABSTAIN") abstain += 1;
    if (r.end_to_end_accuracy) correct += 1;
  });
  const total = rows.length || 1;

  $("count-pass").textContent = pass;
  $("count-regen").textContent = regen;
  $("count-abstain").textContent = abstain;
  $("bar-pass").style.width = `${(pass / total) * 100}%`;
  $("bar-regen").style.width = `${(regen / total) * 100}%`;
  $("bar-abstain").style.width = `${(abstain / total) * 100}%`;

  $("correct-count").textContent = correct;
  $("incorrect-count").textContent = rows.length - correct;

  const pct = rows.length ? correct / rows.length : 0;
  $("donut-pct").textContent = rows.length ? `${(pct * 100).toFixed(1)}%` : "—";
  const circumference = 2 * Math.PI * 52;
  $("acc-donut-arc").setAttribute("stroke-dasharray",
    `${(pct * circumference).toFixed(2)} ${circumference.toFixed(2)}`);
}

function renderCommentary(report) {
  const agg = report.aggregate || {};
  const rows = report.per_query || [];

  const e2e = pickAggregate(agg, "end_to_end_accuracy");
  const f1 = pickAggregate(agg, "claim_f1");
  const mrr = pickAggregate(agg, "retrieval_mrr");
  const prec = pickAggregate(agg, "retrieval_precision_at_5");
  const rec = pickAggregate(agg, "retrieval_recall_at_5");
  const vsr = pickAggregate(agg, "verification_success_rate");

  let pass = 0, regen = 0, abstain = 0;
  rows.forEach((r) => {
    const d = (r.decision || "").toUpperCase();
    if (d === "PASS") pass += 1;
    else if (d === "REGEN") regen += 1;
    else if (d === "ABSTAIN") abstain += 1;
  });

  const lines = [];
  lines.push(`<p>The pipeline was run on <b>${rows.length}</b> synthetic queries. <b>${pass}</b> returned an answer (PASS), <b>${regen}</b> were regenerated, and <b>${abstain}</b> were refused (ABSTAIN). The verification gate is the source of these decisions — its threshold drives the PASS / REGEN / ABSTAIN split.</p>`);

  if (e2e != null) lines.push(`<p><b>End-to-end accuracy</b> is ${fmtPct(e2e)} — i.e. the fraction of queries whose final answer matched the gold reference. This is the top-line number for the pipeline.</p>`);
  if (f1 != null) lines.push(`<p><b>Claim F1</b> is ${fmtPct(f1)}: the harmonic mean of how precisely the pipeline splits answers into atomic claims and how many of those claims it correctly verifies.</p>`);
  if (mrr != null && rec != null && prec != null) {
    lines.push(`<p><b>Retrieval</b> — Recall@5: ${fmtPct(rec)}, Precision@5: ${fmtPct(prec)}, MRR: ${fmtPct(mrr)}. The high recall combined with low precision means the retriever surfaces the right document almost always but also returns irrelevant ones; the cross-encoder reranker (BGE-Reranker-v2) filters most of those out before NLI verification.</p>`);
  }
  if (vsr != null) lines.push(`<p><b>Verification success rate</b> is ${fmtPct(vsr)}: the fraction of claims the DeBERTa-v3 NLI verifier returned a usable verdict on (supported / contradicted / unknown). A high VSR is required before the gate can produce a confident PASS.</p>`);
  lines.push(`<p class="muted small">All numbers are computed by <code>scripts/evaluate.py</code> against <code>data/metrics.json</code>. Detection metrics (ROC-AUC, HRR, CCR, VSR) require <code>scripts/evaluate_full.py</code>.</p>`);

  $("commentary").innerHTML = lines.join("");
}

let ALL_ROWS = [];
let DATASETS = [];
let ACTIVE_SLUG = "";

function renderDatasetSelector() {
  const sel = $("dataset-select");
  if (!sel) return;
  sel.innerHTML = "";
  if (!DATASETS.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "no datasets registered";
    sel.appendChild(opt);
    return;
  }
  DATASETS.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d.slug;
    opt.textContent = d.available
      ? `${d.label} (${d.report?.n_queries || 0} queries)`
      : `${d.label} — not run yet`;
    if (d.slug === ACTIVE_SLUG) opt.selected = true;
    sel.appendChild(opt);
  });
}

function renderComparisonTable() {
  const head = $("comparison-head");
  const body = $("comparison-body");
  const title = $("comparison-title");
  if (!head || !body) return;
  if (!DATASETS.length) {
    head.innerHTML = "<th>Metric</th>";
    body.innerHTML = `<tr><td colspan="2" class="muted small">No datasets.</td></tr>`;
    return;
  }
  // Build header row: Metric | <each dataset> | Δ
  head.innerHTML = "<th>Metric</th>" +
    DATASETS.map((d) => `<th>${escapeHTML(d.label)}${d.available ? "" : " <span class='muted small'>(missing)</span>"}</th>`).join("") +
    "<th>Δ</th>";

  // Build body: fetch comparison data and render
  fetch("/api/evaluation/comparison").then((r) => r.json()).then((data) => {
    const rows = data.rows || [];
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="${DATASETS.length + 2}" class="muted small">No metrics yet — run <code>scripts/evaluate*.py</code>.</td></tr>`;
      return;
    }
    body.innerHTML = rows.map((row) => {
      const cells = DATASETS.map((d) => {
        const v = row[d.slug];
        const cls = (v == null) ? "muted small" :
                    v >= 0.75 ? "cell-good" :
                    v >= 0.5  ? "cell-mid"  : "cell-bad";
        return `<td class="${cls}">${v == null ? "—" : fmtPct(v)}</td>`;
      }).join("");
      const delta = row.delta;
      const dcls = (delta == null) ? "muted small" :
                   delta >= 0    ? "cell-good"  : "cell-bad";
      const dsign = delta == null ? "—" : (delta > 0 ? `+${(delta*100).toFixed(1)}%` : `${(delta*100).toFixed(1)}%`);
      return `<tr><td>${escapeHTML(row.metric)}</td>${cells}<td class="${dcls}">${dsign}</td></tr>`;
    }).join("");
    if (title) title.textContent = `Cross-dataset comparison (${DATASETS.length} datasets)`;
  }).catch(() => {
    body.innerHTML = `<tr><td colspan="${DATASETS.length + 2}" class="muted small">Comparison failed.</td></tr>`;
  });
}

async function loadDatasets() {
  const r = await fetch("/api/evaluation/datasets");
  if (!r.ok) return [];
  const data = await r.json();
  DATASETS = data.datasets || [];
  if (data.default_slug) ACTIVE_SLUG = data.default_slug;
  renderDatasetSelector();
  renderComparisonTable();
  return DATASETS;
}

async function refresh() {
  setDot("", "loading…");
  if (!DATASETS.length) await loadDatasets();
  const qs = ACTIVE_SLUG ? `?dataset=${encodeURIComponent(ACTIVE_SLUG)}` : "";
  const r = await fetch("/api/accuracy" + qs);
  if (!r.ok) {
    setDot("err", `error ${r.status}`);
    return;
  }
  const data = await r.json();
  ALL_ROWS = data.per_query || [];
  renderOverview(data);
  renderDecisions(data);
  renderAggregateGrid("grid-base", BASE_METRICS, data.headline || {});
  if (Object.keys(data.headline || {}).some((k) => k.startsWith("detection_") || k.includes("reduction") || k.includes("claim_correction"))) {
    $("full-section-title").style.display = "";
    $("grid-full").style.display = "";
    renderAggregateGrid("grid-full", FULL_METRICS, data.headline || {});
  }
  renderTable(ALL_ROWS);
  renderCommentary(data);
  // Build metric bars from the raw /api/evaluation endpoint so we get the on-the-fly
  // aggregates too (headline only includes a subset of keys).
  const raw = await fetch("/api/evaluation" + qs);
  if (raw.ok) {
    const evalData = await raw.json();
    renderMetricBars(evalData.aggregate || {});
  } else {
    renderMetricBars(data.headline || {});
  }
  setDot("ok", "live");
}

document.addEventListener("DOMContentLoaded", () => {
  $("btn-refresh").addEventListener("click", refresh);
  $("row-filter").addEventListener("input", () => renderTable(ALL_ROWS));
  $("row-decision").addEventListener("change", () => renderTable(ALL_ROWS));
  const sel = $("dataset-select");
  if (sel) sel.addEventListener("change", () => {
    ACTIVE_SLUG = sel.value;
    refresh();
  });
  refresh();
});
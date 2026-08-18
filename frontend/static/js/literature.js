// Literature comparison page — talks to /api/literature/* on the Flask server.

const $ = (id) => document.getElementById(id);

function escapeHTML(s) {
  return (s ?? "").toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

async function fetchJSON(url) {
  const res = await fetch(url, { headers: { "Content-Type": "application/json" } });
  let data;
  try { data = await res.json(); } catch { data = {}; }
  return { ok: res.ok, status: res.status, data };
}

// ---- Headline card -------------------------------------------------------

function winnerClass(winner) {
  if (!winner) return "";
  if (winner.startsWith("✓")) return "win";
  if (winner.startsWith("≈")) return "tie";
  if (winner.startsWith("✗")) return "loss";
  return "";
}

function renderWinList(targetId, items) {
  const ul = $(targetId);
  if (!ul) return;
  if (!items.length) {
    ul.innerHTML = `<li class="muted small">None in this category.</li>`;
    return;
  }
  ul.innerHTML = items.map((v) => `
    <li>
      <div class="dim-name">${escapeHTML(stripBold(v.dimension))}</div>
      <div class="dim-best">best: ${escapeHTML(v.best_paper)} — ${escapeHTML(stripBold(v.best_score))}</div>
      <div class="dim-best">ours: ${escapeHTML(stripBold(v.our_score))}</div>
    </li>
  `).join("");
}

function stripBold(s) {
  // The paper uses **…**; render as plain text.
  return (s || "").replace(/\*\*/g, "");
}

async function loadHeadline() {
  const r = await fetchJSON("/api/literature/headline");
  if (!r.ok) return;
  const data = r.data || {};
  const tally = data.tally || {};

  // Tally bar.
  const win = tally.win || 0;
  const tie = tally.tie || 0;
  const loss = tally.loss || 0;
  const total = win + tie + loss || 1;
  const bar = $("tally-bar");
  if (bar) {
    bar.innerHTML = `
      <div class="seg-win"  style="width:${(win/total*100).toFixed(1)}%"></div>
      <div class="seg-tie"  style="width:${(tie/total*100).toFixed(1)}%"></div>
      <div class="seg-loss" style="width:${(loss/total*100).toFixed(1)}%"></div>
    `;
  }
  const meta = $("headline-meta");
  if (meta) meta.innerHTML = `<b>${win} wins</b> · ${tie} at par · ${loss} trails &nbsp;·&nbsp; ${total} evaluation dimensions · ${data.methods_compared?.length ?? 11} papers compared`;
  const blurb = $("verdict-blurb");
  if (blurb && tally.verdict) blurb.innerHTML = escapeHTML(stripBold(tally.verdict));
  const wc = $("win-count");  if (wc) wc.textContent = win;
  const tc = $("tie-count");  if (tc) tc.textContent = tie;
  const lc = $("loss-count"); if (lc) lc.textContent = loss;

  renderWinList("win-list",  data.wins || []);
  renderWinList("tie-list",  data.ties || []);
  renderWinList("loss-list", data.losses || []);
}

// ---- Bar chart ------------------------------------------------------------

function tagClass(tag) {
  if (!tag) return "";
  if (tag.includes("Best"))   return "win";
  if (tag.includes("At par")) return "tie";
  if (tag.includes("Tunable"))return "tie";
  return "loss";
}

async function loadBars() {
  const r = await fetchJSON("/api/literature/headline");
  if (!r.ok) return;
  const data = r.data || {};
  const bars = $("lit-bars");
  if (!bars) return;
  const rows = data.visual_headline || [];
  if (!rows.length) {
    bars.innerHTML = `<div class="muted small">No headline chart entries.</div>`;
    return;
  }
  // Determine the maximum absolute value across all rows so the bars
  // share a consistent scale.
  const max = Math.max(...rows.flatMap((r) => [Math.abs(r.best || 0), Math.abs(r.ours || 0)]));
  bars.innerHTML = rows.map((row) => {
    const cls = tagClass(row.tag);
    const widthPct = (v) => max > 0 ? `${(Math.abs(v) / max * 50).toFixed(1)}%` : "0%";
    return `
      <div class="lit-bar-row ${cls}">
        <span class="lit-bar-name">${escapeHTML(row.metric)}</span>
        <span class="lit-bar-track">
          <span class="lit-bar-fill best" style="width:${widthPct(row.best)}"></span>
          <span class="lit-bar-fill ours" style="width:${widthPct(row.ours)}; left:${widthPct(row.best)}"></span>
        </span>
        <span class="lit-bar-legend">${fmt(row.best, 3)} vs ${fmt(row.ours, 3)}</span>
        <span class="lit-bar-tag">${escapeHTML(row.tag)}</span>
      </div>
    `;
  }).join("");
}

// ---- Side-by-side table ---------------------------------------------------

let _litCsv = null;

function renderTable(categoryFilter) {
  const thead = $("lit-thead");
  const tbody = $("lit-tbody");
  if (!thead || !tbody || !_litCsv) return;
  const methods = _litCsv.method_columns || [];
  const rows = _litCsv.rows || [];

  thead.innerHTML = `
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th class="col-ours">Ours</th>
      ${methods.map((m) => `<th>${escapeHTML(m)}</th>`).join("")}
    </tr>
  `;

  const filtered = categoryFilter
    ? rows.filter((r) => (r.category || "").toLowerCase() === categoryFilter.toLowerCase())
    : rows;

  // For each row, find the highest other numeric value to flag wins.
  tbody.innerHTML = filtered.map((row) => {
    const oursNum = row.ours_numeric;
    const otherNums = Object.entries(row.values)
      .map(([k, v]) => ({ k, n: v.num }))
      .filter((x) => x.n !== null);
    const bestOther = otherNums.length ? Math.max(...otherNums.map((x) => x.n)) : null;
    let cellCls = "";
    if (oursNum !== null && oursNum !== undefined && bestOther !== null) {
      if (oursNum >= bestOther) cellCls = "cell-win";
      else if (oursNum < bestOther * 0.85) cellCls = "cell-loss";
      else cellCls = "cell-tie";
    }
    return `
      <tr>
        <td class="cat-tag">${escapeHTML(row.category)}</td>
        <td class="metric-name">${escapeHTML(row.metric)}</td>
        <td class="col-ours">${escapeHTML(row.ours)}</td>
        ${methods.map((m) => {
          const v = row.values[m] || { raw: "—", num: null };
          return `<td>${escapeHTML(v.raw)}</td>`;
        }).join("")}
      </tr>
    `;
  }).join("") || `<tr><td colspan="${3 + methods.length}" class="muted small">No rows.</td></tr>`;

  const rc = $("row-count");
  const cc = $("col-count");
  if (rc) rc.textContent = filtered.length;
  if (cc) cc.textContent = methods.length + 1;  // +1 for "Ours"
}

async function loadTable() {
  const r = await fetchJSON("/api/literature/comparison");
  if (!r.ok) return;
  _litCsv = r.data || {};
  // Populate category filter.
  const select = $("category-filter");
  if (select) {
    const cats = Array.from(new Set((_litCsv.rows || []).map((r) => r.category).filter(Boolean)));
    const opts = ['<option value="">All categories</option>']
      .concat(cats.map((c) => `<option value="${escapeHTML(c)}">${escapeHTML(c)}</option>`));
    select.innerHTML = opts.join("");
    select.onchange = () => renderTable(select.value);
  }
  renderTable("");
}

// ---- Talking points ------------------------------------------------------

async function loadTalkingPoints() {
  // We don't have a dedicated endpoint, so we render static talking points
  // sourced from the why_we_win.md section 6.
  const target = $("talking-points");
  if (!target) return;
  const points = [
    {
      q: "How is your work different from SHROOM?",
      a: "Every SHROOM system is a black-box classifier on a single output. Our pipeline is end-to-end: it generates, retrieves, verifies, regenerates, and calibrates. The SHROOM leaderboard tracks accuracy; we track accuracy + Δ + mitigation lift."
    },
    {
      q: "How is your work different from Dale 2023?",
      a: "Dale scores N-best candidates from a fixed generator with MC-dropout. We rewrite the answer in place using only verified evidence. Dale is +0.56 (LaBSE); we are +0.71. Dale needs a white-box MT model; ours is backend-pluggable."
    },
    {
      q: "How is your work different from Conformal-RAG?",
      a: "Conformal-RAG rejects via 1−α coverage but does not rewrite. We rewrite + abstain. Conformal-RAG is unable to handle RLHF-aligned hallucinations at 95% coverage; we achieve 85% VSR via per-claim NLI."
    },
    {
      q: "What is the strongest single result?",
      a: "The mitigation Δ of +0.67 to +0.75 over the no-mitigation baseline — the largest single improvement in the entire cited literature."
    },
    {
      q: "What are the limitations?",
      a: "(a) No formal conformal coverage guarantee yet (future work). (b) Single QA domain; MT/DM/PG expansion is future work. (c) Live eval is at 200 rows per dataset for time; HaluEval/RAGTruth at scale is the canonical benchmark."
    },
  ];
  target.innerHTML = `<ul>${points.map((p) => `
    <li>
      <div class="tp-q">${escapeHTML(p.q)}</div>
      <div class="tp-a">${escapeHTML(p.a)}</div>
    </li>
  `).join("")}</ul>`;
}

// ---- Init ----------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  const refreshBtn = $("btn-refresh");
  if (refreshBtn) refreshBtn.addEventListener("click", () => {
    loadHeadline(); loadBars(); loadTable();
  });
  loadHeadline();
  loadBars();
  loadTable();
  loadTalkingPoints();
});
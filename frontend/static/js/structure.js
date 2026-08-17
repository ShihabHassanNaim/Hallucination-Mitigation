// Project structure renderer.
const $ = (id) => document.getElementById(id);

const EXT_ICON = {
  py: "py", yaml: "yaml", yml: "yaml", md: "md", json: "json",
  html: "html", htm: "html", css: "css", js: "js", ts: "js", txt: "txt",
};

function fmtSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function escapeHTML(s) {
  return (s ?? "").toString().replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function renderNode(node, filter) {
  if (!node.children || !node.children.length) {
    return `<div class="entry file ${matchesFilter(node, filter) ? "" : "tree-hidden"}">
      <span class="icon ${EXT_ICON[node.ext] || ""}">·</span>
      <span class="name">${escapeHTML(node.name)}</span>
      <span class="meta">${fmtSize(node.size || 0)}</span>
    </div>`;
  }
  const filteredChildren = node.children.filter(c => matchesFilter(c, filter));
  const visible = matchesFilter(node, filter) || filteredChildren.length > 0;
  const open = !filter ? "open" : "";
  const counts = `${node.n_files || 0} files${node.n_dirs ? `, ${node.n_dirs} dirs` : ""}`;
  return `<details class="${visible ? "" : "tree-hidden"}" ${open}>
    <summary>
      <span class="entry">
        <span class="icon dir">▸</span>
        <span class="name"><b>${escapeHTML(node.name)}</b></span>
        <span class="badge">${counts}</span>
      </span>
    </summary>
    ${node.children.map(c => renderNode(c, filter)).join("")}
  </details>`;
}

function matchesFilter(node, f) {
  if (!f) return true;
  const q = f.toLowerCase();
  if (node.name.toLowerCase().includes(q)) return true;
  if (!node.children) return false;
  return node.children.some(c => matchesFilter(c, q));
}

async function loadTree() {
  const depth = $("depth").value || 4;
  const include_hidden = $("hidden").checked ? "1" : "0";
  const r = await fetch(`/api/structure?depth=${depth}&include_hidden=${include_hidden}`);
  if (!r.ok) { $("tree").textContent = "Failed to load"; return; }
  const data = await r.json();
  window._lastTree = data.tree;
  window._lastMeta = data;
  $("root-line").textContent = `root: ${data.root}`;
  $("summary").textContent =
    `${data.total_files} files · ${fmtSize(data.total_size_bytes)} · depth ${data.depth}`;
  applyFilter();
}

function applyFilter() {
  if (!window._lastTree) return;
  const filter = $("filter").value.trim();
  $("tree").innerHTML = renderNode(window._lastTree, filter);
}

document.addEventListener("DOMContentLoaded", () => {
  $("btn-refresh").addEventListener("click", loadTree);
  $("depth").addEventListener("change", loadTree);
  $("hidden").addEventListener("change", loadTree);
  $("filter").addEventListener("input", applyFilter);
  loadTree();
});
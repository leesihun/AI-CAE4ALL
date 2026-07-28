function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

/**
 * Design Parameters has no preview data of its own — it is a placeholder for
 * whichever input/output names arrive via `condition_names`/`feature_names`
 * (set when a real dataset is wired into the pipeline with "Use in
 * pipeline"). Until that happens the table is honestly blank rather than
 * showing a decorative graphic that implies data which isn't there.
 */
export function parametersTableGraphic(node, compact = false) {
  try {
    const table = JSON.parse(node?.config?.parameter_table || "null");
    if (table && Array.isArray(table.columns) && Array.isArray(table.rows) && table.columns.length) {
      const visibleColumns = compact ? table.columns.slice(0, 2) : table.columns;
      const visibleRows = compact ? table.rows.slice(0, 2) : table.rows;
      const header = `<th>Sample</th>${visibleColumns.map(column => `<th>${escapeHtml(column.name || column.id)}</th>`).join("")}`;
      const rows = visibleRows.map((row, index) => `<tr><td>${escapeHtml(row.sample_label || row.sample_id || index + 1)}</td>${visibleColumns.map(column => `<td>${escapeHtml(row.values?.[column.id] || "")}</td>`).join("")}</tr>`).join("");
      const more = compact && (table.rows.length > visibleRows.length || table.columns.length > visibleColumns.length)
        ? `<tr class="parameters-table-more"><td colspan="${visibleColumns.length + 1}">+${Math.max(0, table.rows.length - visibleRows.length)} rows · +${Math.max(0, table.columns.length - visibleColumns.length)} columns</td></tr>`
        : "";
      return `<table class="parameters-table" aria-label="Dataset-aligned design parameters"><thead><tr>${header}</tr></thead><tbody>${rows}${more}</tbody></table>`;
    }
  } catch {
    // Fall through to the legacy name-only preview.
  }
  const inputs = String(node?.config?.condition_names || "").split(",").map(item => item.trim()).filter(Boolean);
  const outputs = String(node?.config?.feature_names || "").split(",").map(item => item.trim()).filter(Boolean);
  const rowCount = Math.max(1, inputs.length, outputs.length);
  const visibleRows = compact ? Math.min(3, rowCount) : rowCount;
  const rows = Array.from({ length: visibleRows }, (unused, index) =>
    `<tr><td>${escapeHtml(inputs[index] || "")}</td><td>${escapeHtml(outputs[index] || "")}</td></tr>`
  ).join("");
  const overflow = compact && rowCount > visibleRows ? `<tr class="parameters-table-more"><td colspan="2">+${rowCount - visibleRows} more</td></tr>` : "";
  return `<table class="parameters-table" aria-label="Design parameters input/output"><thead><tr><th>Input</th><th>Output</th></tr></thead><tbody>${rows}${overflow}</tbody></table>`;
}

export function previewGraphic(kind, seed = 0, large = false) {
  const width = large ? 680 : 220;
  const height = large ? 410 : 80;
  const offset = Number(seed) % 5;
  if (kind === "field") {
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Field prediction preview">
      <defs><linearGradient id="field-${seed}" x1="0" x2="1"><stop stop-color="#2759a7"/><stop offset=".34" stop-color="#32b5c3"/><stop offset=".62" stop-color="#d9e56c"/><stop offset=".84" stop-color="#f09b38"/><stop offset="1" stop-color="#c23d3e"/></linearGradient></defs>
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#22332e"}"/>
      <path d="${large ? "M110 248 225 105 507 144 570 222 447 247 405 326 187 318Z M207 218 265 165 328 186 281 240Z" : "M24 55 64 19 171 27 199 48 153 54 137 73 53 71Z M62 50 82 34 110 39 94 55Z"}" fill="url(#field-${seed})" stroke="rgba(255,255,255,.7)" stroke-width="${large ? 2 : .8}" fill-rule="evenodd"/>
      <g opacity=".4" stroke="#fff" stroke-width="${large ? 1.2 : .45}">${large ? '<path d="M110 248 225 105 281 240 187 318M225 105l103 81 179-42M328 186l77 140M281 240l166 7M187 318l218 8M507 144l-60 103 123-25"/>' : '<path d="M24 55 64 19 94 55 53 71M64 19l46 20 61-12M110 39l27 34M94 55l59-1M53 71l84 2M171 27l-18 27 46-6"/>'}</g>
    </svg>`;
  }
  if (kind === "latent") {
    const yMid = large ? 205 : 40;
    const scale = large ? 2.7 : 1;
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Hierarchical latent model preview">
      <defs><linearGradient id="latent-${seed}" x1="0" x2="1"><stop stop-color="#ae6a7c"/><stop offset=".52" stop-color="#75588e"/><stop offset="1" stop-color="#3f718b"/></linearGradient></defs>
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#f1eaf0"}"/>
      <g transform="translate(${large ? 42 : 13},0)">
        ${[0,1,2,3].map((item) => {
          const x = (large ? 34 : 10) + item * (large ? 103 : 35);
          const h = (20 + item * 10) * scale;
          return `<rect x="${x}" y="${yMid - h/2}" width="${large ? 61 : 21}" height="${h}" rx="${large ? 11 : 4}" fill="url(#latent-${seed})" opacity="${.58 + item*.1}"/>`;
        }).join("")}
        <path d="${large ? "M416 205h74M520 205h72" : "M150 40h25M184 40h21"}" stroke="#7d5c91" stroke-width="${large ? 4 : 1.5}" stroke-linecap="round"/>
        <circle cx="${large ? 505 : 179}" cy="${yMid}" r="${large ? 20 : 7}" fill="#fff" stroke="#9f6478" stroke-width="${large ? 4 : 1.5}"/>
        <path d="${large ? "m499 197 14 8-14 8Z" : "m177 37 5 3-5 3Z"}" fill="#9f6478"/>
      </g>
    </svg>`;
  }
  if (kind === "training") {
    const points = large ? "46,335 110,290 172,255 238,226 301,192 365,168 428,147 492,119 555,103 632,82" : "10,68 31,58 52,50 74,44 96,35 118,30 141,24 165,20 188,15 213,11";
    const points2 = large ? "46,360 110,320 172,285 238,260 301,229 365,207 428,183 492,166 555,141 632,126" : "10,73 31,64 52,57 74,51 96,45 118,39 141,34 165,30 188,25 213,21";
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Training metrics preview">
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#f1f4ef"}"/>
      <g stroke="#dbe0da" stroke-width="1">${[1,2,3,4].map(i => `<path d="M0 ${i*height/5}H${width}"/>`).join("")}</g>
      <polyline points="${points}" fill="none" stroke="#19715e" stroke-width="${large ? 5 : 2}" stroke-linecap="round" stroke-linejoin="round"/>
      <polyline points="${points2}" fill="none" stroke="#b97838" stroke-width="${large ? 4 : 1.6}" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`;
  }
  if (kind === "parameters") {
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Parameter controls preview">
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#f6ede5"}"/>
      <g stroke="#d7c4b4" stroke-width="${large ? 8 : 3}" stroke-linecap="round">
        ${[0,1,2,3].map(i => `<path d="M${large ? 92 : 28} ${large ? 112+i*62 : 20+i*15}H${large ? 590 : 193}"/>`).join("")}
      </g>
      ${[0,1,2,3].map(i => `<circle cx="${(large ? 195 : 64) + ((i+offset)%4)*(large ? 80 : 27)}" cy="${large ? 112+i*62 : 20+i*15}" r="${large ? 10 : 3.5}" fill="#fff" stroke="#b0713f" stroke-width="${large ? 4 : 1.3}"/>`).join("")}
    </svg>`;
  }
  if (kind === "checkpoint") {
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Saved model preview">
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#efebf2"}"/>
      ${[0,1,2,3,4,5].map(i => {
        const x = large ? 86+i*86 : 28+i*31;
        const h = large ? 110+(i%3)*35 : 26+(i%3)*9;
        return `<rect x="${x}" y="${height/2-h/2}" width="${large ? 45 : 15}" height="${h}" rx="${large ? 7 : 2}" fill="${i%2 ? "#8d70a2" : "#6f5290"}" opacity="${.65+i*.05}"/>`;
      }).join("")}
    </svg>`;
  }
  if (kind === "candidates") {
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="CAD candidate gallery preview">
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#eef1e8"}"/>
      ${[0,1,2].map(i => {
        const x = large ? 70+i*205 : 18+i*68;
        const y = large ? 95+(i%2)*25 : 16+(i%2)*4;
        const size = large ? 130 : 42;
        return `<path d="M${x+size*.18} ${y+size*.1} ${x+size*.76} ${y} ${x+size} ${y+size*.55} ${x+size*.69} ${y+size} ${x+size*.13} ${y+size*.81} ${x} ${y+size*.36}Z" fill="${["#88a640","#a7b84d","#6f9238"][i]}" stroke="rgba(255,255,255,.75)" stroke-width="${large ? 4 : 1.2}"/>`;
      }).join("")}
    </svg>`;
  }
  if (kind === "export") {
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Export manifest preview">
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#eef0ed"}"/>
      <rect x="${large ? 175 : 58}" y="${large ? 60 : 9}" width="${large ? 330 : 104}" height="${large ? 290 : 62}" rx="${large ? 12 : 4}" fill="#fff" stroke="#cfd4cf" stroke-width="${large ? 3 : 1}"/>
      ${[0,1,2,3].map(i => `<path d="M${large ? 225 : 72} ${large ? 122+i*48 : 23+i*12}H${large ? 455 : 148}" stroke="#74827c" stroke-width="${large ? 5 : 1.7}" stroke-linecap="round"/>`).join("")}
    </svg>`;
  }
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Geometry sample preview">
    <defs><linearGradient id="mesh-${seed}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#7db3b4"/><stop offset="1" stop-color="#315e6d"/></linearGradient></defs>
    <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#e8ede7"}"/>
    <path d="${large ? "M98 292 275 112 249 352M171 64l80 179 112-185M251 243l115 61 165 29M363 58l7 176 211-105M390 158l141 175M290 139l-41 213M171 64l192-6" : "M28 62 91 26 84 75M58 17l26 39 37-38M84 56l40 5 62 7M121 18l2 40 70-22M128 39l58 29M95 34 84 75M58 17l63 1"}" fill="none" stroke="url(#mesh-${seed})" stroke-width="${large ? 8 : 2.8}" stroke-linecap="round" stroke-linejoin="round"/>
    ${[0,1,2,3,4,5].map(i => `<circle cx="${large ? 115+i*82 : 31+i*31}" cy="${large ? 250-((i+offset)%3)*58 : 58-((i+offset)%3)*14}" r="${large ? 8 : 2.5}" fill="#e2efe9" stroke="#44766a" stroke-width="${large ? 3 : 1}"/>`).join("")}
  </svg>`;
}

export function nodeVisualLabel(spec) {
  if (spec.modelId === "simulgenvae") return "hierarchical latent + condition mapping";
  if (spec.visual === "field") return "click field samples";
  if (spec.visual === "candidates") return "click candidate gallery";
  if (spec.visual === "training") return "training + validation";
  if (spec.visual === "dataset") return "samples + geometry + fields";
  if (spec.visual === "checkpoint") return "saved model + .pth";
  return spec.sampleLabel;
}

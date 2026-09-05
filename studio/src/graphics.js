import { MESH_PREVIEW, FIELD_PREVIEW } from "./previewdata.js";

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

// The Studio's field ramp, shared with the artifact viewer.
const FIELD_STOPS = ["#2759a7", "#32b5c3", "#d9e56c", "#f09b38", "#c23d3e"];

function fieldRampStops(id) {
  return `<linearGradient id="${id}" x1="0" y1="1" x2="0" y2="0">${
    FIELD_STOPS.map((color, index) =>
      `<stop offset="${(index / (FIELD_STOPS.length - 1)).toFixed(2)}" stop-color="${color}"/>`).join("")
  }</linearGradient>`;
}

/**
 * A loss curve shaped like a real one — fast early drop into a noisy plateau, with
 * validation sitting slightly above training — rather than the two smooth straight
 * lines the old preview drew. Deterministic, so previews never flicker on re-render.
 */
function lossCurve(count, amplitude, floor, decay, phase) {
  return Array.from({ length: count }, (unused, index) => {
    const t = index / (count - 1);
    const envelope = Math.exp(-decay * t);
    return floor + amplitude * envelope * (1 + 0.09 * Math.sin(index * 1.9 + phase) * envelope);
  });
}

function polyline(values, x0, y0, plotWidth, plotHeight, maximum) {
  return values.map((value, index) => {
    const x = x0 + (index / (values.length - 1)) * plotWidth;
    const y = y0 + plotHeight - (value / maximum) * plotHeight;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
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

/**
 * The placeholder a block shows before it has produced anything.
 *
 * Deliberately not a chart: empty axes plus one line of text, so it cannot be
 * mistaken for a measurement at a glance the way a drawn curve can.
 */
function emptyPreview(width, height, large, message) {
  const pad = large ? 46 : 16;
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="No result yet: ${message}">
    <rect width="${width}" height="${height}" fill="${large ? "#fbfcfa" : "#f4f6f2"}"/>
    <path d="M${pad} ${pad * 0.5}V${height - pad * 0.7}H${width - pad}" fill="none" stroke="#dfe4dd" stroke-width="${large ? 1.6 : 1}"/>
    <text x="${width / 2}" y="${height / 2 + (large ? 6 : 3)}" text-anchor="middle" fill="#9aa8a1"
          font-size="${large ? 17 : 10}" font-style="italic">${message}</text>
  </svg>`;
}

export function previewGraphic(kind, seed = 0, large = false, hasEvidence = true) {
  const width = large ? 680 : 220;
  const height = large ? 410 : 80;
  const offset = Number(seed) % 5;
  // A block with no run behind it must not show a curve that looks like its
  // result. The training and latent previews are illustrations of what the block
  // produces; drawn on a card whose own evidence line reads "No run linked",
  // they were indistinguishable from a finished training run.
  if (!hasEvidence && (kind === "training" || kind === "latent")) {
    return emptyPreview(width, height, large, kind === "latent" ? "no reconstruction yet" : "no run yet");
  }
  if (kind === "field") {
    // A real ex9 plasticity field rasterised through the Studio ramp, plus the
    // colour bar that makes it readable as a contour plot instead of decoration.
    const bar = large ? 16 : 7;
    const inset = large ? 26 : 7;
    const plotWidth = width - inset * 2 - bar - (large ? 46 : 12);
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Predicted field preview from ex9 plasticity">
      <defs>${fieldRampStops(`fieldbar-${seed}`)}</defs>
      <rect width="${width}" height="${height}" fill="${large ? "#101d19" : "#16241f"}"/>
      <image href="${large ? FIELD_PREVIEW.large : FIELD_PREVIEW.small}" x="${inset}" y="${inset}"
             width="${plotWidth}" height="${height - inset * 2}" preserveAspectRatio="xMidYMid meet"/>
      <rect x="${width - inset - bar}" y="${inset}" width="${bar}" height="${height - inset * 2}"
            rx="${large ? 4 : 2}" fill="url(#fieldbar-${seed})" stroke="rgba(255,255,255,.35)" stroke-width="${large ? 1.2 : .5}"/>
      ${large ? `<text x="${width - inset - bar - 8}" y="${inset + 10}" fill="rgba(255,255,255,.82)" font-size="13" text-anchor="end">${FIELD_PREVIEW.max}</text>
        <text x="${width - inset - bar - 8}" y="${height - inset - 2}" fill="rgba(255,255,255,.82)" font-size="13" text-anchor="end">${FIELD_PREVIEW.min}</text>` : ""}
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
    // The old preview drew two straight lines rising to the right, which is the
    // wrong shape for a loss and read as a generic "chart" sticker. This is a real
    // decay-into-plateau with validation above training, on proper axes.
    const padLeft = large ? 54 : 20;
    const padBottom = large ? 40 : 14;
    const padTop = large ? 22 : 7;
    const padRight = large ? 22 : 8;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;
    const count = large ? 46 : 26;
    const train = lossCurve(count, 1, 0.075, 3.4, 0);
    const valid = lossCurve(count, 1.06, 0.135, 3.0, 1.7);
    const maximum = Math.max(...train, ...valid) * 1.08;
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Training and validation loss preview">
      <rect width="${width}" height="${height}" fill="${large ? "#fbfcfa" : "#f4f6f2"}"/>
      <g stroke="#dfe4dd" stroke-width="${large ? 1.2 : .7}">
        ${[0, 1, 2, 3].map(i => {
          const y = padTop + (i / 3) * plotHeight;
          return `<path d="M${padLeft} ${y.toFixed(1)}H${width - padRight}"/>`;
        }).join("")}
      </g>
      <path d="M${padLeft} ${padTop}V${padTop + plotHeight}H${width - padRight}" fill="none" stroke="#b6bfb8" stroke-width="${large ? 1.6 : .9}"/>
      <polyline points="${polyline(valid, padLeft, padTop, plotWidth, plotHeight, maximum)}" fill="none"
                stroke="#b97838" stroke-width="${large ? 3.4 : 1.5}" stroke-linecap="round" stroke-linejoin="round"/>
      <polyline points="${polyline(train, padLeft, padTop, plotWidth, plotHeight, maximum)}" fill="none"
                stroke="#19715e" stroke-width="${large ? 4 : 1.8}" stroke-linecap="round" stroke-linejoin="round"/>
      ${large ? `<text x="${padLeft - 10}" y="${padTop + 6}" fill="#6c7872" font-size="13" text-anchor="end">loss</text>
        <text x="${padLeft}" y="${height - 12}" fill="#6c7872" font-size="13">epoch</text>
        <g font-size="13"><rect x="${width - 190}" y="${padTop + 4}" width="16" height="4" rx="2" fill="#19715e"/>
        <text x="${width - 168}" y="${padTop + 11}" fill="#4c5a54">train</text>
        <rect x="${width - 116}" y="${padTop + 4}" width="16" height="4" rx="2" fill="#b97838"/>
        <text x="${width - 94}" y="${padTop + 11}" fill="#4c5a54">validation</text></g>` : ""}
    </svg>`;
  }
  if (kind === "parity") {
    // Predicted vs. ground truth around y=x — what an evaluation block actually
    // produces. It previously borrowed the training-loss curve, which is a
    // different quantity entirely.
    const pad = large ? 34 : 9;
    const box = Math.min(width, height) - pad * 2;
    const x0 = (width - box) / 2;
    const y0 = (height - box) / 2;
    const dots = Array.from({ length: large ? 90 : 34 }, (unused, i) => {
      const t = (i + 0.5) / (large ? 90 : 34);
      const spread = 0.045 * Math.sin(i * 2.7) + 0.028 * Math.cos(i * 1.3);
      const px = t;
      const py = Math.min(0.99, Math.max(0.01, t + spread));
      return `<circle cx="${(x0 + px * box).toFixed(1)}" cy="${(y0 + (1 - py) * box).toFixed(1)}" r="${large ? 4 : 1.6}" fill="#3f6f91" opacity=".72"/>`;
    }).join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Predicted versus ground-truth parity preview">
      <rect width="${width}" height="${height}" fill="${large ? "#fbfcfa" : "#f4f6f2"}"/>
      <rect x="${x0}" y="${y0}" width="${box}" height="${box}" fill="none" stroke="#d5dcd6" stroke-width="${large ? 1.4 : .8}"/>
      <path d="M${x0} ${y0 + box}L${x0 + box} ${y0}" stroke="#9aa8a1" stroke-width="${large ? 2 : 1}" stroke-dasharray="${large ? "7 5" : "3 2"}"/>
      ${dots}
      ${large ? `<text x="${x0}" y="${y0 + box + 24}" fill="#6c7872" font-size="13">ground truth</text>
        <text x="${x0 - 8}" y="${y0 + 4}" fill="#6c7872" font-size="13" text-anchor="end">predicted</text>` : ""}
    </svg>`;
  }
  if (kind === "ranking") {
    // Model comparison is a ranking, not a time series.
    const rows = large ? 5 : 4;
    const pad = large ? 30 : 8;
    const rowH = (height - pad * 2) / rows;
    const labelW = large ? 96 : 30;
    const maxW = width - pad * 2 - labelW;
    const values = [1, 0.78, 0.61, 0.44, 0.29];
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Model ranking preview">
      <rect width="${width}" height="${height}" fill="${large ? "#fbfcfa" : "#f4f6f2"}"/>
      ${Array.from({ length: rows }, (unused, i) => {
        const y = pad + i * rowH + rowH * 0.18;
        const h = rowH * 0.62;
        return `<rect x="${pad + labelW}" y="${y.toFixed(1)}" width="${(maxW * values[i]).toFixed(1)}" height="${h.toFixed(1)}"
                  rx="${large ? 4 : 2}" fill="${i === 0 ? "#19715e" : "#7ba394"}" opacity="${i === 0 ? 1 : 0.55 + (rows - i) * 0.08}"/>
                <rect x="${pad + labelW - (large ? 74 : 24)}" y="${(y + h * 0.28).toFixed(1)}" width="${large ? 62 : 20}" height="${large ? 6 : 3}"
                  rx="3" fill="#c3ccc6"/>`;
      }).join("")}
    </svg>`;
  }
  if (kind === "pareto") {
    // Objective space with a highlighted non-dominated front.
    const pad = large ? 34 : 9;
    const w = width - pad * 2;
    const h = height - pad * 2;
    const frontCount = large ? 7 : 5;
    const front = Array.from({ length: frontCount }, (unused, i) => {
      const t = i / (frontCount - 1);
      return [pad + t * w, pad + h - (1 - Math.pow(t, 1.7)) * h * 0.92];
    });
    const cloud = Array.from({ length: large ? 44 : 20 }, (unused, i) => {
      const t = ((i * 37) % 100) / 100;
      const lift = ((i * 61) % 100) / 100;
      const x = pad + t * w;
      const y = pad + h - (1 - Math.pow(t, 1.7)) * h * 0.92 + lift * h * 0.42;
      return `<circle cx="${x.toFixed(1)}" cy="${Math.min(y, pad + h).toFixed(1)}" r="${large ? 4 : 1.7}" fill="#9aa8a1" opacity=".6"/>`;
    }).join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Pareto front preview">
      <rect width="${width}" height="${height}" fill="${large ? "#fbfcfa" : "#f4f6f2"}"/>
      <path d="M${pad} ${pad}V${pad + h}H${pad + w}" fill="none" stroke="#c3ccc6" stroke-width="${large ? 1.6 : .9}"/>
      ${cloud}
      <polyline points="${front.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")}" fill="none"
                stroke="#b76b2a" stroke-width="${large ? 3 : 1.4}" stroke-dasharray="${large ? "8 5" : "3 2"}"/>
      ${front.map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${large ? 6 : 2.6}" fill="#b76b2a" stroke="#fff" stroke-width="${large ? 2 : .8}"/>`).join("")}
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
    // Reads as "a trained network on disk". The old version was six bars of
    // arbitrary height, which said nothing about what a .pth actually holds.
    const layers = [4, 6, 6, 3];
    const gap = large ? 150 : 48;
    const x0 = large ? 130 : 42;
    const r = large ? 11 : 3.6;
    const nodes = layers.map((n, li) => Array.from({ length: n }, (unused, ni) => ({
      x: x0 + li * gap,
      y: height / 2 + (ni - (n - 1) / 2) * (large ? 62 : 17)
    })));
    const links = nodes.slice(0, -1).flatMap((layer, li) =>
      layer.flatMap(a => nodes[li + 1].map(bNode =>
        `<path d="M${a.x} ${a.y}L${bNode.x} ${bNode.y}"/>`))).join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Saved model checkpoint preview">
      <rect width="${width}" height="${height}" fill="${large ? "#f4f1f7" : "#efebf2"}"/>
      <g stroke="#b4a0c6" stroke-width="${large ? .9 : .35}" opacity=".75" fill="none">${links}</g>
      ${nodes.flat().map(node => `<circle cx="${node.x}" cy="${node.y}" r="${r}" fill="#7a5c96" stroke="#fff" stroke-width="${large ? 2.4 : .8}"/>`).join("")}
    </svg>`;
  }
  if (kind === "candidates") {
    // Three *distinct* load-bearing brackets with mounting holes — a generated
    // design family. The old version drew the same irregular blob three times.
    const scale = large ? 1 : 0.322;
    const step = large ? 205 : 68;
    const x0 = large ? 68 : 17;
    const yBase = large ? 96 : 17;
    const webs = ["M0 0h118v34H74l-8 96H0Z", "M0 0h118v40H62l14 90H0Z", "M0 0h108v30H70l4 100H0Z"];
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Generated CAD candidate family preview">
      <rect width="${width}" height="${height}" fill="${large ? "#f2f5ec" : "#eef1e8"}"/>
      ${[0, 1, 2].map(i => `<g transform="translate(${x0 + i * step} ${yBase}) scale(${scale})">
        <path d="${webs[i]}" fill="${["#88a640", "#a7b84d", "#6f9238"][i]}" stroke="rgba(255,255,255,.8)"
              stroke-width="${large ? 4 : 9}" stroke-linejoin="round"/>
        <circle cx="26" cy="17" r="9" fill="${large ? "#f2f5ec" : "#eef1e8"}"/>
        <circle cx="26" cy="${100 + i * 6}" r="9" fill="${large ? "#f2f5ec" : "#eef1e8"}"/>
      </g>`).join("")}
    </svg>`;
  }
  if (kind === "export") {
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Export manifest preview">
      <rect width="${width}" height="${height}" fill="${large ? "transparent" : "#eef0ed"}"/>
      <rect x="${large ? 175 : 58}" y="${large ? 60 : 9}" width="${large ? 330 : 104}" height="${large ? 290 : 62}" rx="${large ? 12 : 4}" fill="#fff" stroke="#cfd4cf" stroke-width="${large ? 3 : 1}"/>
      ${[0,1,2,3].map(i => `<path d="M${large ? 225 : 72} ${large ? 122+i*48 : 23+i*12}H${large ? 455 : 148}" stroke="#74827c" stroke-width="${large ? 5 : 1.7}" stroke-linecap="round"/>`).join("")}
    </svg>`;
  }
  // Default: the real ex9 mesh. The previous version drew seven random strokes and
  // six floating dots, which looked like scribble rather than a discretised body.
  const mesh = large ? MESH_PREVIEW.large : MESH_PREVIEW.small;
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Mesh sample preview from ex9 plasticity">
    <rect width="${width}" height="${height}" fill="${large ? "#f4f7f4" : "#eef2ee"}"/>
    <path d="${mesh.outline}" fill="#dcebe4" stroke="none"/>
    <path d="${mesh.d}" fill="none" stroke="#6d9d92" stroke-width="${large ? .9 : .5}" opacity=".9"/>
    <path d="${mesh.outline}" fill="none" stroke="#2f5f54" stroke-width="${large ? 3 : 1.3}" stroke-linejoin="round"/>
  </svg>`;
}

export function nodeVisualLabel(spec) {
  if (spec.modelId === "simulgenvae") return "hierarchical latent + condition mapping";
  if (spec.visual === "field") return "click field samples";
  if (spec.visual === "candidates") return "click candidate gallery";
  if (spec.visual === "training") return "training + validation";
  if (spec.visual === "parity") return "predicted vs ground truth";
  if (spec.visual === "ranking") return "ranked model metrics";
  if (spec.visual === "pareto") return "objective space + Pareto front";
  if (spec.visual === "dataset") return "mesh samples + fields";
  if (spec.visual === "checkpoint") return "saved model + .pth";
  return spec.sampleLabel;
}

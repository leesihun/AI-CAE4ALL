import { escapeHtml } from "./dom.js";

/**
 * Minimal Markdown rendering for the Docs workspace.
 *
 * The pane used to print the raw source into a <pre>: the docs index showed 42
 * links and not one of them was clickable, headings read as "## " and tables as
 * pipe soup. This covers what the repository's documents actually use --
 * headings, fenced and inline code, lists, tables, blockquotes, rules,
 * bold/italic and links.
 *
 * Everything is escaped before any markup is added, so no document content can
 * inject HTML. Only this function's own tags survive.
 *
 * Link handling is what makes the docs navigable:
 *  - an absolute http(s) link opens in a new tab;
 *  - a repository-relative link resolves against the current document's own
 *    directory and is tagged `data-doc-link`, which the Docs workspace opens in
 *    the same pane;
 *  - a bare `#anchor` becomes plain text rather than a link that goes nowhere.
 */
export function renderMarkdown(text, path = "") {
  const baseDir = String(path).replace(/\\/g, "/").split("/").slice(0, -1).join("/");

  const resolve = href => {
    const clean = String(href).split("#")[0];
    if (!clean) return "";
    const stack = [];
    `${baseDir}/${clean}`.split("/").forEach(part => {
      if (!part || part === ".") return;
      if (part === "..") stack.pop();
      else stack.push(part);
    });
    return stack.join("/");
  };

  const inline = raw => {
    let out = escapeHtml(raw);
    out = out.replace(/`([^`]+)`/g, (match, code) => `<code>${code}</code>`);
    out = out.replace(/\*\*([^*]+)\*\*/g, (match, bold) => `<strong>${bold}</strong>`);
    out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, (match, before, italic) => `${before}<em>${italic}</em>`);
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, label, href) => {
      if (/^(https?:)?\/\//.test(href)) {
        return `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer noopener">${label}</a>`;
      }
      if (href.startsWith("#")) return label;
      const target = resolve(href);
      return target ? `<a href="#" data-doc-link="${escapeHtml(target)}">${label}</a>` : label;
    });
    return out;
  };

  const lines = String(text).split(/\r?\n/);
  const html = [];
  let listType = "";
  let inFence = false;
  let fence = [];
  let table = [];

  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = "";
  };

  const flushFence = () => {
    html.push(`<pre class="doc-code">${escapeHtml(fence.join("\n"))}</pre>`);
    fence = [];
  };

  const closeTable = () => {
    if (!table.length) return;
    const cells = row => row.replace(/^\||\|$/g, "").split("|").map(cell => cell.trim());
    const header = cells(table[0]);
    // A separator row (|---|---|) is layout, not data.
    const body = table.slice(table.length > 1 && /^[\s|:-]+$/.test(table[1]) ? 2 : 1);
    const head = header.map(cell => `<th>${inline(cell)}</th>`).join("");
    const rows = body.map(row => `<tr>${cells(row).map(cell => `<td>${inline(cell)}</td>`).join("")}</tr>`).join("");
    html.push(`<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`);
    table = [];
  };

  lines.forEach(line => {
    if (/^\s*```/.test(line)) {
      if (inFence) flushFence();
      else { closeList(); closeTable(); }
      inFence = !inFence;
      return;
    }
    if (inFence) { fence.push(line); return; }

    if (/^\s*\|.*\|\s*$/.test(line)) { closeList(); table.push(line.trim()); return; }
    closeTable();

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      // Shifted one level down: the workspace already supplies an <h1>.
      const level = Math.min(6, heading[1].length + 1);
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      return;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const wanted = bullet ? "ul" : "ol";
      if (listType !== wanted) { closeList(); html.push(`<${wanted}>`); listType = wanted; }
      html.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      return;
    }
    closeList();

    if (/^\s*>\s?/.test(line)) {
      html.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
      return;
    }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { html.push("<hr>"); return; }
    if (!line.trim()) return;
    html.push(`<p>${inline(line)}</p>`);
  });

  if (inFence && fence.length) flushFence();
  closeList();
  closeTable();
  return html.join("");
}

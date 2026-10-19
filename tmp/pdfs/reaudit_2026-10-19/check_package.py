from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import re
import runpy
import zipfile
from unittest.mock import patch
import fitz

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / 'paper/HI-MGN-hybrid'
NAME = 'HI_MGN_HYBRID_Introduction_Methodology'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
evidence = json.loads((P / 'CURRENT_VERIFICATION.json').read_text('utf8'))
qa = json.loads((P / 'FINAL_QA.json').read_text('utf8'))
visual = json.loads((P / qa['visual_review_record']).read_text('utf8'))
report = {'date': '2026-10-19 Asia/Seoul', 'pdf_sha256': sha(P / f'{NAME}.pdf')}
report['source_hash_mismatches'] = [s['path'] for s in evidence['sources'] if sha(ROOT / s['path']) != s['sha256']]
report['manuscript_hash_mismatches'] = [s['path'] for s in evidence['manuscript_sources'] if sha(P / s['path']) != s['sha256']]
report['pdf_matches_build_and_records'] = report['pdf_sha256'] == sha(P / f'build/{NAME}.pdf') == qa['pdf_sha256'] == visual['pdf_sha256']
report['cached_render_hash_mismatches'] = [n for n, h in visual['rendered_page_hashes'].items() if sha(P / 'build/rendered' / n) != h]
with zipfile.ZipFile(P / f'{NAME}_source.zip') as z:
    report['zip_crc_failure'] = z.testzip()
    report['zip_entries'] = len(z.namelist())
    report['zip_live_mismatches'] = [n for n in z.namelist() if not (P / n).is_file() or z.read(n) != (P / n).read_bytes()]
captured = {}
def capture(path, data, *args, **kwargs):
    captured[path.name] = data
    return len(data)
with patch.object(Path, 'write_text', capture), contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(P / 'verify_manuscript.py'))
    runpy.run_path(str(P / 'make_markdown.py'))
report['fresh_structural_checks'] = json.loads(captured['CURRENT_VERIFICATION.json'])['checks']
report['markdown_reproduces_exactly'] = captured[f'{NAME}.md'] == (P / f'{NAME}.md').read_text('utf8')
report['helper_syntax_errors'] = []
for f in P.glob('*.py'):
    try:
        ast.parse(f.read_text('utf8'))
    except SyntaxError as e:
        report['helper_syntax_errors'].append({'file': f.name, 'error': str(e)})
report['missing_local_markdown_targets'] = []
for f in P.glob('*.md'):
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', f.read_text('utf8')):
        target = target.split('#', 1)[0]
        if target and not re.match(r'\w+://', target) and not (f.parent / target).exists():
            report['missing_local_markdown_targets'].append({'file': f.name, 'target': target})
with fitz.open(P / f'{NAME}.pdf') as pdf:
    report['pages'] = len(pdf)
    report['replacement_characters'] = []
    report['text_outside_page'] = []
    report['fresh_render_differs_from_cache'] = []
    report['pdf_unresolved_placeholders'] = []
    for i, page in enumerate(pdf, 1):
        text = page.get_text()
        if '\ufffd' in text:
            report['replacement_characters'].append(i)
        if re.search(r'\?\?|\b(?:TODO|TBD|PLACEHOLDER)\b', text):
            report['pdf_unresolved_placeholders'].append(i)
        for block in page.get_text('blocks'):
            if not page.rect.contains(fitz.Rect(block[:4])):
                report['text_outside_page'].append(i)
        fresh = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        cached = fitz.Pixmap(str(P / f'build/rendered/page_{i:02d}.png'))
        if fresh.samples != cached.samples:
            report['fresh_render_differs_from_cache'].append(i)
log = (P / f'build/{NAME}.log').read_text('utf8', errors='replace')
report['tex_fatal_or_reference_warnings'] = [line for line in log.splitlines() if re.search(r'Overfull|undefined|^! |Emergency stop|Fatal error', line)]
report['tex_underfull_count'] = log.count('Underfull')
report['unchanged_sources_after_check'] = all(sha(P / s['path']) == s['sha256'] for s in evidence['manuscript_sources'])
report['backup_checks'] = []
for directory in ['before_2026-10-17_hybrid_novelty_revision', 'before_2026-10-19_recheck']:
    b = P / 'audit' / directory
    manifest = json.loads((b / 'BACKUP_MANIFEST.json').read_text('utf8'))
    report['backup_checks'].append({'directory': directory, 'files': len(manifest['files']), 'mismatches': [item['path'] for item in manifest['files'] if sha(b / item['path']) != item['sha256']]})
Path(__file__).with_name('automated_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(report, ensure_ascii=False, indent=2))

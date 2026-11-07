"""Verify saved audit integrity and reproduce the main and CFE tallies."""
import collections
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda name: json.loads((root / name).read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

for item in read('final-input-manifest.json'):
    assert digest(Path(item['file'])) == item['sha256'], item['file']
assert digest(root / 'blind-verdicts.jsonl') == read('blind-lock.json')['sha256']
for name in ('cfe-independent-verdicts', 'cfe-baseline-verdicts'):
    assert digest(root / (name + '.json')) == (root / (name + '.sha256')).read_text().split()[0]

rows = read('grades-confirmed-corrections-only.json')
assert len(rows) == 2256
assert len({(r['model'], r['run'], r['id']) for r in rows}) == len(rows)
summary = read('audit-summary.json')
manual = {(r['run'], r['id']): r for r in read('manual-verdicts.json')}
assert len(manual) == 312
wrong = [r for r in rows if r['model'] == 'f5' and r['scope'] == 'main' and r['machine_grade'] == 'wrong']
assert len(wrong) == 227
assert all((r['run'], r['id']) in manual for r in wrong)

for model in ('f5', 'untouched'):
    for scope, key in (('main', 'main_confirmed_corrections_only'), ('cfe', 'cfe_manual')):
        selected = [r for r in rows if r['model'] == model and r['scope'] == scope]
        counts = dict(collections.Counter(r['confirmed_corrections_grade'] for r in selected))
        expected = summary[model][key]
        assert counts == expected['counts'], (model, scope, counts)
        assert len(selected) == expected['n']
        assert sum(r['gen_tokens'] for r in selected) == expected['tokens']
        assert len({r['id'] for r in selected if r['confirmed_corrections_grade'] == 'right'}) == expected['at_least_once_right']
        assert all(r['confirmed_corrections_grade'] == 'cut' for r in selected if r['finish'] == 'length')

blind = read('blind-comparison.json')
expected_strata = collections.Counter({(run, grade): n for run in ('g', 's1', 's2', 's3') for grade, n in (('right', 20), ('wrong', 5))})
assert collections.Counter((r['run'], r['machine_grade']) for r in blind) == expected_strata
for item in read('output-manifest.json'):
    assert digest(root / item['file']) == item['sha256'], item['file']
print('Source hashes, locked verdicts, review coverage, saved artifact hashes, and main/CFE tallies verified.')

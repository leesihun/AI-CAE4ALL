"""`opt_budget 0` screening and `opt_stress_margin 0` (no stress constraint)."""

import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_loop.loop import calibrate, select_best  # noqa: E402
from design_loop.problem import MassObjective  # noqa: E402
from design_loop.surrogate import surrogate_baseline_population  # noqa: E402
from inference_profiles.optimize import (  # noqa: E402
    _plot,
    _screen_progress,
    _screening_summary,
    _summarize,
    _write_report,
    _write_screening_table,
)


def _fea(mass, peak=100e6, disp=1e-3, uz=None, nodes=5000):
    return {'mass': mass, 'volume': mass / 4000.0, 'num_nodes': nodes,
            'peak_von_mises': peak, 'max_von_mises': peak * 1.1,
            'max_displacement': disp, 'vertical_displacement': uz,
            'cases': {'vertical': {'peak_von_mises': peak, 'max_von_mises': peak * 1.1,
                                   'max_displacement': disp}}}


def _ok(index, mass, **kwargs):
    return {'ok': True, 'index': index, 'x': [float(index)], 'fea': _fea(mass, **kwargs)}


def test_stress_margin_zero_drops_the_stress_constraint():
    records = [_ok(0, 1.0, peak=100e6), _ok(1, 2.0, peak=300e6), _ok(2, 3.0, peak=200e6)]
    limits = calibrate(records, stress_margin=0)
    assert limits['stress_allow'] is None and limits['stress_constraint'] is False
    assert limits['stress_range'] == [100e6, 300e6]
    assert calibrate(records)['stress_allow'] == 200e6
    with pytest.raises(ValueError):
        calibrate(records, stress_margin=-0.5)

    objective = MassObjective(limits['mass_ref'], limits['stress_allow'], limits['disp_allow'],
                              vertical_disp_allow=0.15e-3)
    # Far past any stress, within the deflection limit: feasible, scored on mass alone.
    score, penalty = objective(_fea(1.0, peak=5e9, uz=0.1e-3))
    assert penalty['feasible'] and penalty['stress_violation'] == 0.0
    assert score == pytest.approx(1.0 / limits['mass_ref'])
    assert objective(_fea(1.0, uz=0.2e-3))[1]['feasible'] is False


class _Generator:
    """A 'mesh' carries its design's first coordinate; listed calls crash."""

    def __init__(self, crash_at=()):
        self.crash_at, self.calls = set(crash_at), 0

    def bounds(self):
        return np.zeros(2), np.ones(2)

    def generate(self, x):
        call, self.calls = self.calls, self.calls + 1
        if call in self.crash_at:
            raise RuntimeError('decoder blew up')
        return SimpleNamespace(tag=float(x[0])), {'call': call}


class _Surrogate:
    """One analyze_batch per chunk; listed calls die like a native crash."""

    def __init__(self, fail_calls=()):
        self.calls, self.fail_calls, self.last_errors = [], set(fail_calls), {}

    def analyze_batch(self, meshes):
        self.calls.append(len(meshes))
        if len(self.calls) - 1 in self.fail_calls:
            raise RuntimeError('native call died\nTraceback ...')
        return [{'mass': m.tag, 'volume': 1.0, 'num_nodes': 10, 'peak_von_mises': 1e8,
                 'max_von_mises': 1e8, 'max_displacement': 1e-3,
                 'vertical_displacement': m.tag * 1e-3,
                 'cases': {'vertical': {'peak_von_mises': 1e8}}} for m in meshes]


def test_chunked_screen_is_the_one_call_population_reported_per_chunk():
    whole = surrogate_baseline_population(_Generator(), _Surrogate(), size=7, seed=3,
                                          verbose=False)
    progress, surrogate = [], _Surrogate()
    chunked = surrogate_baseline_population(
        _Generator(), surrogate, size=7, seed=3, verbose=False, batch_size=3,
        on_chunk=lambda records, total: progress.append((len(records), total)))
    assert surrogate.calls == [3, 3, 1]
    assert progress == [(3, 7), (6, 7), (7, 7)]
    # The designs are drawn up front: chunking changes the calls, not the population.
    assert [r['x'] for r in chunked] == [r['x'] for r in whole]
    assert [r['fea']['mass'] for r in chunked] == [r['fea']['mass'] for r in whole]
    assert [r['index'] for r in chunked] == list(range(7))
    # batch_size 0 is one call, as None is.
    one = _Surrogate()
    surrogate_baseline_population(_Generator(), one, size=5, verbose=False, batch_size=0)
    assert one.calls == [5]


def test_a_crashing_design_and_a_failed_later_chunk_do_not_end_the_screen(capsys):
    surrogate = _Surrogate(fail_calls={1})
    records = surrogate_baseline_population(_Generator(crash_at={0}), surrogate, size=6,
                                            verbose=False, batch_size=2)
    # The crashed design never reaches the surrogate.
    assert surrogate.calls == [1, 2, 2]
    assert records[0]['ok'] is False and records[0]['error'] == 'RuntimeError: decoder blew up'
    assert records[1]['ok']
    assert [r['ok'] for r in records[2:4]] == [False, False]
    assert records[2]['error'].startswith('RuntimeError: native call died')
    assert all(r['ok'] for r in records[4:])
    assert 'chunk 2-3 failed: RuntimeError: native call died' in capsys.readouterr().out


def test_a_broken_surrogate_stops_the_screen_at_its_first_chunk():
    with pytest.raises(RuntimeError, match='native call died'):
        surrogate_baseline_population(_Generator(), _Surrogate(fail_calls={0}), size=4,
                                      verbose=False, batch_size=2)


def test_progress_line_tracks_the_lightest_design_meeting_the_vertical_limit(capsys):
    records = [_ok(0, 0.3, uz=0.2e-3), _ok(1, 0.5, uz=0.1e-3),
               {'ok': False, 'index': 2, 'error': 'x'}]
    _screen_progress(0.15e-3, started=0.0)(records, 10)
    line = capsys.readouterr().out
    assert 'screened 3/10 | solved 2 | u_z <= 0.150 mm: 1' in line
    assert 'lightest meeting it #1 0.5000 kg (u_z 0.1000 mm)' in line


def _screen():
    limits = {'population': 4, 'mass_ref': 0.6, 'mass_range': [0.3, 0.9],
              'stress_allow': None, 'stress_constraint': False,
              'stress_range': [50e6, 400e6], 'disp_allow': 1e-3,
              'disp_range': [1e-3, 1e-3], 'vertical_disp_allow': 0.15e-3}
    objective = MassObjective(0.6, None, 1e-3, vertical_disp_allow=0.15e-3)
    baseline = [_ok(0, 0.3, uz=0.2e-3), _ok(1, 0.5, uz=0.12e-3, peak=400e6),
                _ok(2, 0.7, uz=0.1e-3), _ok(3, 0.9, uz=0.05e-3),
                {'ok': False, 'index': 4, 'x': [4.0],
                 'error': 'RuntimeError: no zero crossing\nmore'}]
    for record in baseline:
        if record['ok']:
            record['score'], record['penalty'] = objective(record['fea'])
    return limits, objective, baseline


def test_screen_delivers_the_lightest_design_meeting_the_limits(tmp_path):
    limits, _, baseline = _screen()
    delivered = select_best([r for r in baseline if r['ok']])
    # The lightest meeting 0.15 mm, not the lightest overall; with no stress
    # constraint its 400 MPa peak does not disqualify it.
    assert delivered['index'] == 1
    typical = baseline[2]

    screen = _screening_summary(baseline, delivered, typical, limits, 64, 50.0)
    assert (screen['designs'], screen['solved'], screen['feasible']) == (5, 4, 3)
    assert screen['meeting_vertical_limit'] == 3
    assert screen['delivered_id'] == 'screen_0001' and screen['delivered_feasible']
    assert screen['typical_id'] == 'screen_0002'
    assert screen['seconds_per_design'] == 10.0 and screen['batch_size'] == 64

    path = _write_screening_table(str(tmp_path), baseline, delivered, typical, limits,
                                  'surrogate')
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    assert [r['id'] for r in rows] == [f'screen_{i:04d}' for i in range(5)]
    assert rows[0]['vertical_limit_met'] == 'false' and rows[1]['vertical_limit_met'] == 'true'
    assert rows[1]['vertical_displacement_mm'] == '0.12' and rows[1]['num_nodes'] == '5000'
    assert (rows[1]['role'], rows[1]['path']) == ('delivered', 'optimized.stl')
    assert (rows[2]['role'], rows[2]['path']) == ('typical', 'typical.stl')
    assert rows[0]['role'] == '' and rows[0]['path'] == ''
    # A failed design keeps its reason and no numbers -- never a mass of 0.
    assert rows[4]['mass_kg'] == '' and rows[4]['score'] == ''
    assert rows[4]['error'] == 'RuntimeError: no zero crossing'

    # Delivered and typical the same design: one row, one resolvable path.
    _write_screening_table(str(tmp_path), baseline, delivered, delivered, limits, 'surrogate')
    with open(path, newline='', encoding='utf-8') as fh:
        row = list(csv.DictReader(fh))[1]
    assert (row['role'], row['path']) == ('delivered;typical', 'optimized.stl')


def test_screen_summary_and_report_have_no_baseline_column(tmp_path):
    limits, objective, baseline = _screen()
    delivered, typical = baseline[1], baseline[2]
    verified = {'ok': True, 'fea': _fea(0.51, uz=0.13e-3, peak=420e6)}
    typical_v = {'ok': True, 'fea': _fea(0.72, uz=0.11e-3)}
    evaluator = SimpleNamespace(failures={}, objective=objective)
    material = SimpleNamespace(name='Ti', E=1.0, nu=0.3, rho=1.0, yield_stress=1.0)
    summary = _summarize(limits, delivered, None, verified, [1.0], delivered['score'],
                         [], [], baseline, evaluator, material, ('vertical',),
                         SimpleNamespace(length_scale=0.1, stress_percentile=99.5), 60.0,
                         delivered, typical_v, 'surrogate', verify_evaluations=2)
    v = summary['verified']
    assert 'baseline' not in v and 'mass_change_pct' not in v
    assert v['vertical_limit_met'] is True and v['optimized_penalty']['feasible'] is True
    assert 'vs_typical' in v and summary['total_evaluations'] == 5 + 2

    summary.update(search_improved_on_baseline=None,
                   screening=_screening_summary(baseline, delivered, typical, limits, 64, 50.0))
    _write_report(str(tmp_path), summary)
    report = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert '## Screening' in report and 'No search was run (`opt_budget 0`)' in report
    assert 'in chunks of 64 per native call' in report
    assert '4 of 5 screened designs completed the chain' in report
    assert 'stress constraint **off** (`opt_stress_margin 0`)' in report
    assert 'Delivered design (lightest screened design meeting the limits)' in report
    assert '`screen_0001` re-generated at the verification resolution' in report
    assert 'vertical-case max |u_z| 0.1300 mm (limit met)' in report
    assert 'Against the median-mass member' in report
    assert 'did not improve' not in report and 'Population median mass' not in report


def test_screen_plot_needs_no_search_and_no_stress_allowable(tmp_path):
    pytest.importorskip('matplotlib')
    limits, _, baseline = _screen()
    _plot(str(tmp_path), baseline, [], [], limits, 'surrogate', delivered_index=1)
    assert (tmp_path / 'convergence.png').stat().st_size > 0

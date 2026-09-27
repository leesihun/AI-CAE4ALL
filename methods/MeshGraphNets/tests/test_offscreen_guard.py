"""The periodic PNG must never be able to kill a training run.

An X-only VTK (the PyPI vtk<9.4 wheels) calls abort() when it opens a render
window with no DISPLAY. That is a SIGABRT, not an exception: an HI-MGN run on a
headless server died with exit code -6 right after its first test_interval,
though every plotting call sits inside a try/except. The guard probes one
render in a child process, and these tests pin its decisions without VTK.
"""
import subprocess
import types

import numpy as np

from general_modules import mesh_utils_fast as m


def _reset(monkeypatch, display=None, platform='linux', os_name='posix'):
    monkeypatch.setattr(m, '_OFFSCREEN_OK', None)
    monkeypatch.setattr(m.os, 'name', os_name)
    monkeypatch.setattr(m.sys, 'platform', platform)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY'):
        monkeypatch.delenv(key, raising=False)
    if display:
        monkeypatch.setenv('DISPLAY', display)


def test_child_abort_disables_rendering(monkeypatch):
    _reset(monkeypatch)
    calls = []

    def aborted(*args, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(returncode=-6)

    monkeypatch.setattr(subprocess, 'run', aborted)
    assert m.offscreen_rendering_available() is False
    assert m.offscreen_rendering_available() is False
    assert len(calls) == 1, 'the probe must run once per process, not per picture'


def test_child_success_enables_rendering(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=0))
    assert m.offscreen_rendering_available() is True


def test_display_skips_the_probe(monkeypatch):
    _reset(monkeypatch, display=':0')

    def forbidden(*args, **kwargs):
        raise AssertionError('no probe needed with a display')

    monkeypatch.setattr(subprocess, 'run', forbidden)
    assert m.offscreen_rendering_available() is True


def test_plot_returns_false_instead_of_opening_a_window(monkeypatch, tmp_path):
    monkeypatch.setattr(m, '_OFFSCREEN_OK', False)

    def forbidden(*args, **kwargs):
        raise AssertionError('Plotter must not be constructed when rendering is unavailable')

    monkeypatch.setattr(m.pv, 'Plotter', forbidden)
    pos = np.random.default_rng(0).normal(size=(4, 3))
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    vals = np.ones((2, 1))
    out = tmp_path / 'p.png'
    assert m.plot_mesh_comparison(pos, faces, vals, vals, vals, vals, str(out)) is False
    assert not out.exists()

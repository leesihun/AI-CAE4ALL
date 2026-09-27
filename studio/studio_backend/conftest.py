"""Point every Studio runtime path at a throwaway directory for the test run.

`studio_backend.state` builds `STATE = StudioState()` at import time, and that
constructor runs job recovery over `JOB_RUNTIME`: against the real
`studio/runtime/jobs/` it rewrites the records of a live server's jobs. The
tests also save configs, reports and exports. So the three runtime roots are
redirected here, before any test module can import a studio_backend module
that captures them by name (`from studio_backend.paths import RUNTIME_ROOT`).
pytest imports this conftest before collecting the test modules beside it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from studio_backend import paths

_TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="studio-test-runtime-")).resolve()
paths.RUNTIME_ROOT = _TEST_RUNTIME
paths.CONFIG_RUNTIME = _TEST_RUNTIME / "configs"
paths.JOB_RUNTIME = _TEST_RUNTIME / "jobs"


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001 - pytest hook signature
    shutil.rmtree(_TEST_RUNTIME, ignore_errors=True)

"""
Shared pytest setup for the simulator test suite.

- `--slow` runs tests marked @pytest.mark.slow (skipped by default).
- The `record` fixture stores key numbers per test; at the end of the session
  tests/report.json lists every test with its result and numbers.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))

import warp as wp  # noqa: E402

wp.config.quiet = True

_NUMBERS = {}
_RESULTS = {}
_T0 = time.time()


def _plain(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


@pytest.fixture
def record(request):
    store = _NUMBERS.setdefault(request.node.nodeid, {})

    def _rec(**kw):
        store.update({k: _plain(v) for k, v in kw.items()})
    return _rec


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="also run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (run with --slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow test (use --slow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _RESULTS[report.nodeid] = {"result": report.outcome, "duration_s": round(report.duration, 2)}
        if report.outcome == "failed":
            _RESULTS[report.nodeid]["message"] = str(report.longrepr.reprcrash.message
                                                     if hasattr(report.longrepr, "reprcrash") else report.longrepr)[:500]


def pytest_sessionfinish(session, exitstatus):
    try:
        import subprocess
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except Exception:
        smi = []
    out = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(time.time() - _T0, 1),
                    "slow": bool(session.config.getoption("--slow")), "python": platform.python_version(),
                    "warp": wp.__version__, "numpy": np.__version__, "gpu": smi},
           "tests": [{"test": k, **v, "numbers": _NUMBERS.get(k, {})} for k, v in _RESULTS.items()]}
    with open(os.path.join(ROOT, "tests", "report.json"), "w") as f:
        json.dump(out, f, indent=1)

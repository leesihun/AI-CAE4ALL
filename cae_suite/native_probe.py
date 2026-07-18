#!/usr/bin/env python3
"""Invoke a method's native config loader without importing the training entrypoint."""

from __future__ import annotations

import json
import os
import sys
import traceback


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(json.dumps({"ok": False, "error": "usage: native_probe.py <config>"}))
        return 2
    sys.path.insert(0, os.getcwd())
    try:
        from general_modules.load_config import load_config
        config = load_config(argv[0])
        print("__CAE_SUITE_NATIVE_RESULT__" + json.dumps({"ok": True, "keys": sorted(config)}))
        return 0
    except Exception as exc:
        print("__CAE_SUITE_NATIVE_RESULT__" + json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

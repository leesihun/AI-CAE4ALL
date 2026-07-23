#!/usr/bin/env python
"""Stand-alone CLI entrypoint for the AI-CAE4ALL inference bundle.

    python run_inference.py --checkpoint model.pth --input scene.h5 --output out/

Works from a plain checkout of this `inference/` folder, or frozen into
`run_inference.exe` via PyInstaller (see pyinstaller.spec). No other part of
the AI-CAE4ALL repository is required at runtime.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cae_infer.cli import main

if __name__ == "__main__":
    sys.exit(main())

# AI-CAE4ALL port of the SAR package (Lino & Thuerey 2026). Upstream also
# exported ``datasets``, ``plot`` and ``metrics`` (dataset downloaders, vtk
# plotting, POT-based Wasserstein metrics); none of them is used to train or
# sample, so they are not vendored. See ../NOTICE and ../README.md.
from .graph import Graph
from . import transforms, nn
from .loader import DataLoader

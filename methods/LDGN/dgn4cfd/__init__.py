# AI-CAE4ALL port of dgn4cfd (Lino, Pfaff & Thuerey, ICLR 2025: diffusion
# graph nets and their latent variant, the LDGN). Upstream also exported ``datasets``, ``plot`` and
# ``metrics`` (dataset downloaders, matplotlib plotting, POT-based metrics);
# none of them is used to train or sample an LDGN, so they are not vendored.
# See ../NOTICE and ../README.md.
from .graph import Graph
from . import transforms, nn
from .loader import DataLoader

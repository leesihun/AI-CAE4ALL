# AI-CAE4ALL port: only ``mesh`` (MeshCoarsening) is vendored. The adapter in
# general_modules/mesh_dataset.py does what upstream's ScaleEdgeAttr, Copy and
# Cond transforms did for the Wing example (edge scaling, target copy,
# conditioning), so connect/scale/dirichlet/identity/copy/cond are not.
from .mesh import MeshCoarsening

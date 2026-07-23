"""geometry_ingest: CAD/geometry -> shared mesh HDF5 contract (volume + point cloud).

Upstream ingestion tool. Reads STEP/IGES/STL/PLY/OBJ, produces a welded node set
with connectivity, and writes the existing ``data/{id}/{nodal_data, mesh_edge}``
contract so every mesh-consuming method (MeshGraphNets, Transolver,
Neural_Operator) reads it with no conversion step.

Runs standalone (``python -m geometry_ingest.cli``) or through the suite launcher
(``python AI_CAE4ALL_main.py --config …`` with ``model geometry_ingest``). See
README.md.
"""

from . import clean, config, pipeline, readers, to_graph, to_pointcloud, writer

__all__ = [
    "readers", "clean", "to_graph", "to_pointcloud", "writer", "pipeline", "config",
]

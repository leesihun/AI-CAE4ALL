"""Closed-loop geometry optimization on the DeepJEB SDFFlow generator.

generate (SDFFlow) -> mesh (gmsh) -> analyze (tet FEA) -> score -> search (CMA-ES)
"""

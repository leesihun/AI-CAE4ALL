"""Legacy variational-MGN pipeline helpers; not a cHI-MGNflow runtime route.

The copied stage implementation depends on a VAE encoder and reconstruction
losses that cHI-MGNflow does not have. ``CHiMGNFlow_main.py`` therefore rejects
``parallel_mode=model_split`` and never imports the legacy launcher.
"""

from parallelism.partition import partition_stages
from parallelism.profile import BlockEstimate, profile_activation_memory

__all__ = ['partition_stages', 'profile_activation_memory', 'BlockEstimate']

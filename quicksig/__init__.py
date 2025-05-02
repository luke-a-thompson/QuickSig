"""
QuickSig - A fast signature computation library.
"""

from .path_signature import batch_signature_pure_jax as compute_path_signature
from .batch_ops import batch_otimes_pure_jax, batch_seq_otimes_pure_jax, batch_restricted_exp_pure_jax

__version__ = "0.1.0"
__all__ = ["compute_path_signature", "batch_otimes_pure_jax", "batch_seq_otimes_pure_jax", "batch_restricted_exp_pure_jax"]

"""
QuickSig - A fast signature computation library.
"""

from .path_signature import batch_signature_pure_jax as compute_path_signature
from .batch_ops import batch_tensor_product, batch_seq_tensor_product, batch_restricted_exp, batch_cauchy_prod, batch_tensor_log

__version__ = "0.1.0"
__all__ = ["compute_path_signature", "batch_tensor_product", "batch_seq_tensor_product", "batch_restricted_exp", "batch_cauchy_prod", "batch_tensor_log"]

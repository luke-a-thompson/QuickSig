"""
QuickSig - A fast signature computation library using JAX.

This library provides efficient computation of path signatures and log signatures
using JAX for high performance and GPU acceleration.

Main functions:
    get_signature: Compute path signatures
    get_log_signature: Compute log signatures
"""

from .signatures import get_signature_dim, get_log_signature_dim

__version__ = "0.1.0"
__author__ = "Luke Thompson"
__email__ = "luke.thompson@sydney.edu.au"
__license__ = "MIT"

__all__ = ["get_signature", "get_log_signature", "get_signature_dim", "get_log_signature_dim"]

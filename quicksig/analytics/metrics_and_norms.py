"""
This module contains functions for computing metrics and norms of signature levels.
"""

import math


def hurst_to_holder_a(H: float, epsilon: float = 0.01) -> float:
    """Convert Hölder exponent to Holder alpha."""
    if H <= 0:
        raise ValueError(f"H must be positive. Got H={H}.")
    if H > 1 / 2:
        return 1
    alpha = H - epsilon
    if alpha <= 0:
        raise ValueError(f"Hölder exponent (H - ε) must be positive. Got α={alpha}.")
    return alpha


def hurst_to_minimal_signature_depth(H: float, epsilon: float = 0.01) -> int:
    """Convert Hölder exponent to minimal signature depth."""
    alpha = hurst_to_holder_a(H, epsilon)
    return math.floor(1 / alpha)

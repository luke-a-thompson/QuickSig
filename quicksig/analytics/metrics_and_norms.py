"""
This module contains functions for computing metrics and norms of signature levels.
"""

import math


def get_holder_alpha(H: float, epsilon: float = 0.01) -> float:
    if H <= 0:
        raise ValueError(f"H must be positive. Got H={H}.")
    if H > 1 / 2:
        return 1
    alpha = H - epsilon
    if alpha <= 0:
        raise ValueError(f"Hölder exponent (H - ε) must be positive. Got α={alpha}.")
    return alpha


def get_minimal_signature_depth(H: float, epsilon: float = 0.01) -> int:
    alpha = get_holder_alpha(H, epsilon)
    return math.floor(1 / alpha)




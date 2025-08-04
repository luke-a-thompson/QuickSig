import jax
import jax.numpy as jnp
from functools import partial
from .compute_path_signature import compute_path_signature
from .compute_log_signature import compute_log_signature
from typing import Literal
from .signature_types import Signature


@partial(jax.jit, static_argnames=["depth", "mode", "flatten"])
def signature_transform(
    path: jax.Array,
    depth: int,
    mode: Literal["full", "stream", "incremental"] = "full",
    flatten: bool = True,
) -> Signature:
    """
    Compute the signature of a path.

    Args:
        path: The path to compute the signature of.
        depth: The depth of the signature.
        mode:
            "full": compute the full signature.
            "stream": compute a stream of signatures.
            "incremental": compute the signature of each increment.
        flatten: Whether to flatten the signature.
    """
    match mode:
        case "full":
            signature = compute_path_signature(path, depth, mode=mode)
            return flatten_signature(signature, mode=mode) if flatten else signature
        case "stream":
            signature = compute_path_signature(path, depth, mode=mode)
            return flatten_signature(signature, mode=mode) if flatten else signature
        case "incremental":
            signature = compute_path_signature(path, depth, mode=mode)
            return flatten_signature(signature, mode=mode) if flatten else signature
        case _:
            raise ValueError(f"Invalid mode: {mode}")

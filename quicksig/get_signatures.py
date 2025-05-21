import jax
from quicksig.signature import batch_signature
from quicksig.log_signature import batch_log_signature, LogSignatureType
from functools import partial


@partial(jax.jit, static_argnames=["depth", "stream"])
def get_signature(path: jax.Array, depth: int, stream: bool = False) -> jax.Array:
    """
    Compute the signature of a path.
    """
    return batch_signature(path, depth, stream=stream, flatten=True)  # type: ignore


@partial(jax.jit, static_argnames=["depth", "log_signature_type"])
def get_log_signature(path: jax.Array, depth: int, log_signature_type: LogSignatureType) -> jax.Array:
    """
    Compute the log signature of a path.
    """
    return batch_log_signature(path, depth, log_signature_type=log_signature_type)  # type: ignore

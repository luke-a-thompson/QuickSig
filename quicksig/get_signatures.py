import jax
import jax.numpy as jnp
from functools import partial
from quicksig.path_signature import batch_signature
from quicksig.log_signature import batch_log_signature, LogSignatureType


@partial(jax.jit, static_argnames=["depth", "stream"])
def get_signature(path: jax.Array, depth: int, stream: bool = False) -> jax.Array:
    """
    Compute the signature of a path.
    """
    signature = batch_signature(path, depth, stream=stream)  # pyright: ignore[reportReturnType]
    return flatten_signature(signature, stream=stream)


@partial(jax.jit, static_argnames=["depth", "log_signature_type"])
def get_log_signature(path: jax.Array, depth: int, log_signature_type: LogSignatureType) -> jax.Array:
    """
    Compute the log signature of a path.
    """
    log_signature = batch_log_signature(path, depth, log_signature_type=log_signature_type)
    return flatten_signature(log_signature, stream=False)


def flatten_signature(signature: list[jax.Array], stream: bool = False) -> jax.Array:
    return jnp.concatenate(signature, axis=1) if not stream else jnp.concatenate(signature, axis=2)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    path = jax.random.normal(key, shape=(10, 100, 2))  # Stream
    signature: jax.Array = get_signature(path, depth=5)
    print(signature.shape)
    log_signature: jax.Array = get_log_signature(path, depth=5, log_signature_type=LogSignatureType.LYNDON)
    print(log_signature.shape)

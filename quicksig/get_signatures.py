import jax
import jax.numpy as jnp
from functools import partial
from quicksig.path_signature import path_signature
from quicksig.log_signature import path_log_signature
from typing import Literal


@partial(jax.jit, static_argnames=["depth", "stream"])
def get_signature(path: jax.Array, depth: int, stream: bool = False) -> jax.Array:
    """
    Compute the signature of a path.
    """
    signature = path_signature(path, depth, stream=stream)  # pyright: ignore[reportReturnType]
    return flatten_signature(signature, stream=stream)


@partial(jax.jit, static_argnames=["depth", "log_signature_type", "stream"])
def get_log_signature(
    path: jax.Array,
    depth: int,
    log_signature_type: Literal["expanded", "lyndon"],
    stream: bool = False,
) -> jax.Array:
    """
    Compute the log signature of a path.
    """
    log_signature = path_log_signature(path, depth, log_signature_type=log_signature_type, stream=stream)
    return flatten_signature(log_signature, stream=stream)


def flatten_signature(signature: list[jax.Array], stream: bool = False) -> jax.Array:
    return jnp.concatenate(signature, axis=0) if not stream else jnp.concatenate(signature, axis=1)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    path = jax.random.normal(key, shape=(5, 100, 3))  # Stream
    signature: jax.Array = jax.vmap(get_signature, in_axes=(0, None, None))(path, 4, True)
    print(signature.shape)

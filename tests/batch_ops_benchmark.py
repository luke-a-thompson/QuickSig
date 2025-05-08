import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import time
from quicksig.batch_ops import batch_tensor_product, batch_seq_tensor_product


@jax.jit
def batch_otimes_expand(x: jax.Array, y: jax.Array) -> jax.Array:
    """GPU-optimized batched tensor product using expand_dims."""
    xdim = x.ndim
    ydim = y.ndim
    for i in range(ydim - 1):
        x = jnp.expand_dims(x, -1)
    for i in range(xdim - 1):
        y = jnp.expand_dims(y, 1)
    return x * y


def benchmark() -> None:
    # Generate test data for batch tensor product
    batch_size = 1000  # Number of vectors in the batch
    n = 1000  # Dimension of first vector (x)
    m = 1000  # Dimension of second vector (y)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, n))
    y = jax.random.normal(key, (batch_size, m))

    # Warm up JIT
    _ = batch_tensor_product(x, y)
    _ = batch_otimes_expand(x, y)

    # Benchmark einsum version
    start = time.time()
    for _ in range(100):
        _ = batch_tensor_product(x, y)
    einsum_time = time.time() - start

    # Benchmark expand_dims version
    start = time.time()
    for _ in range(100):
        _ = batch_otimes_expand(x, y)
    expand_time = time.time() - start

    print("Batch tensor product results:")
    print(f"Einsum version: {einsum_time:.4f} seconds")
    print(f"Expand_dims version: {expand_time:.4f} seconds")
    print(f"Einsum is {expand_time/einsum_time:.2f}x faster")


if __name__ == "__main__":
    benchmark()

import jax
import jax.numpy as jnp
from functools import partial
from quicksig.batch_ops import batch_restricted_exp_pure_jax, batch_seq_otimes_pure_jax
import lovely_jax as lj

lj.monkey_patch()


@partial(jax.jit, static_argnames=["depth", "stream"])
def batch_signature_pure_jax(path: jax.Array, depth: int, stream: bool = False) -> jax.Array:
    """Computes the path signature of a path up to a given depth. We omit the constant first order signature.

    Args:
        path: a jax.Array of shape (batch_size, seq_len, n_features)
        depth: the depth of the signature to compute
        stream: whether to return a stream of signatures or a the (single) final signature of the whole path.
        For example, if depth=2, stream=True will return a stream of length seq_len-1,
        each element of which is a tensor of shape (batch_size, 2, n_features^3)

    Returns:
        If stream=False:
            jax.Array of shape (batch_size, n_features ** (depth + 1))
        If stream=True:
            jax.Array of shape (batch_size, seq_len - 1, n_features ** (depth + 1))

    """
    assert depth > 0 and isinstance(depth, int), "Depth must be a positive integer"

    batch_size, seq_len, n_features = path.shape

    # Shape: [batch_size, seq_len - 1, n_features]
    path_increments = path[:, 1:] - path[:, :-1]

    # stacked[k].shape: [batch_size, seq_len - 1, n_features ** (k + 1)]
    # A series of lists, where each seq_len element contains the prefix signature (the signature from [0, t] for t in [0, seq_len - 1])
    # If T=100, we have 99 level 1 signatures as we drop the t = 0 increment (constant signature).
    # At first order, the signature at t = 5 would be the sum of the increments from t = 1 to t = 5.
    incremental_signatures = [jnp.cumsum(path_increments, axis=1)]
    if depth == 1:
        if stream:
            # Full time‑series signature: shape (B, T‑1, D)
            return incremental_signatures[0]
        else:
            # Only the end‑point signature, flattened: shape (B, D)
            return incremental_signatures[0][:, -1, :]

    # Tuple of shape: [(batch_size, n_features ** (k + 1)) for k = 0, 1, ..., depth - 1]
    # Example: depth = 2
    #   exp_term[0].shape: [batch_size, n_features]
    #   exp_term[1].shape: [batch_size, n_features, n_features]
    # For the first increment, we compute the tensor restricted exponential \frac{(\Deltax_0)^{\otimes k}}{k!}, \quad \forall k \in [1, depth]
    # Truncated tensor exponential for the first increment (timestep) of k
    first_inc_tensor_exp_terms: tuple[jax.Array, ...] = batch_restricted_exp_pure_jax(path_increments[:, 0, :], depth=depth)

    # Precompute scaled path increments `\delta_{x_t}/2, \delta_{x_t}/3, \dots, \delta_{x_t}/k` for each k in [2, depth]
    # Precalculates the scaling factor for the different depths
    path_increment_divided_list: list[jax.Array] = []
    for k in range(2, depth + 1):
        scaled = path_increments / k
        path_increment_divided_list.append(scaled)
    # Shape: [depth - 1, batch_size, seq_len - 1, n_features]
    path_increment_divided = jnp.stack(path_increment_divided_list, axis=0)

    for depth_index in range(1, depth):
        # Everything up to but not including the current increment + the current increment scaled by k
        current = incremental_signatures[0][:, :-1, :] + path_increment_divided[depth_index - 1, :, 1:, :]

        for j in range(depth_index - 1):
            # Shuffle product, notice that j increases while depth_index-j increases
            current = incremental_signatures[j + 1][:, :-1, :] + batch_seq_otimes_pure_jax(current, path_increment_divided[depth_index - j - 2, :, 1:])
        current = batch_seq_otimes_pure_jax(current, path_increments[:, 1:])

        # Concatenate the first increment (timestep) with the rest of the signature
        first_inc_expanded = jnp.expand_dims(first_inc_tensor_exp_terms[depth_index], axis=1)  # Shape: [batch_size, 1, n_features ** (depth_index + 1)]
        current = jnp.concatenate([first_inc_expanded, current], axis=1)  # Shape: [batch_size, depth_index + 1, n_features ** (depth_index + 1)]

        # The depth-k signature up to time t
        incremental_signatures.append(jnp.cumsum(current, axis=1))

    if not stream:
        return jnp.concatenate([jnp.reshape(c[:, -1], (batch_size, n_features ** (1 + idx))) for idx, c in enumerate(incremental_signatures)], axis=1)
    else:
        return jnp.concatenate([jnp.reshape(r, (batch_size, seq_len - 1, n_features ** (1 + idx))) for idx, r in enumerate(incremental_signatures)], axis=2)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    path = jax.random.normal(key, shape=(1, 100, 3))
    signature = batch_signature_pure_jax(path, depth=5)

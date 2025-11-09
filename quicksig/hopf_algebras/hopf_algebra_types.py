from typing import NamedTuple, NewType
import jax.numpy as jnp

class Forest(NamedTuple):
    """A batch container for a forest of rooted trees.

    Parameters
    - parent: 2D array of shape ``(num_trees, n)`` with dtype ``int32``.
      Each row encodes one rooted tree via its parent array in preorder:
      ``parent[0] == -1`` and for ``i > 0`` we have ``0 <= parent[i] < i``.

    Notes
    - This container is compatible with JAX; the array can be a ``jax.Array``.
    - The number of nodes ``n`` is the same for all trees in the forest.

    Example
    >>> import jax.numpy as jnp
    >>> forest = Forest(parent=jnp.array([[-1, 0, 0]], dtype=jnp.int32))
    >>> forest.parent.shape
    (1, 3)
    """

    parent: jnp.ndarray


MKWForest = NewType("MKWForest", Forest)
BCKForest = NewType("BCKForest", Forest)

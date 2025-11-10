from typing import NamedTuple, NewType
from dataclasses import dataclass
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from quicksig.tensor_ops import cauchy_convolution


class HopfAlgebra(ABC):
    """Abstract Hopf algebra interface sufficient for signature/log-signature workflows."""

    @abstractmethod
    def ambient_dimension(self) -> int:
        """Ambient dimension of the underlying alphabet / features."""
        raise NotImplementedError

    @abstractmethod
    def basis_size(self, level: int) -> int:
        """Number of basis elements at given level (degree = level + 1 for signatures).

        Implementations must define how many coefficients live at each level.
        """
        raise NotImplementedError

    @abstractmethod
    def star(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        """Group law (convolution-star) on truncated coefficients, degree-wise.

        Args:
            a_levels: list of flattened tensors per degree (omits degree 0)
            b_levels: same shape/depth as a_levels
        Returns:
            list of flattened tensors per degree representing a ⋆ b
        """
        raise NotImplementedError

    def zero(self, depth: int, dtype: jnp.dtype) -> list[jax.Array]:
        return [jnp.zeros((self.basis_size(i),), dtype=dtype) for i in range(depth)]


@dataclass(frozen=True)
class ShuffleHopfAlgebra(HopfAlgebra):
    """Shuffle/Tensor Hopf algebra used for path signatures.

    The representation uses per-degree flattened tensors, omitting degree 0.
    """

    _ambient_dimension: int

    def ambient_dimension(self) -> int:
        return int(self._ambient_dimension)

    def basis_size(self, level: int) -> int:
        # level i corresponds to tensors of order (i+1)
        dim = self.ambient_dimension()
        return int(dim ** (level + 1))

    def _unflatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        dim = self.ambient_dimension()
        return [term.reshape((dim,) * (i + 1)) for i, term in enumerate(levels)]

    def _flatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        return [term.reshape(-1) for term in levels]

    def star(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncations must match for star product.")
        dim = self.ambient_dimension()
        depth = len(a_levels)
        a_unflat = self._unflatten_levels(a_levels)
        b_unflat = self._unflatten_levels(b_levels)
        cross_unflat = cauchy_convolution(a_unflat, b_unflat, depth, a_unflat)
        out_unflat = [a + b + c for a, b, c in zip(a_unflat, b_unflat, cross_unflat)]
        return self._flatten_levels(out_unflat)


def convolution_star(hopf: HopfAlgebra, a: list[jax.Array], b: list[jax.Array]) -> list[jax.Array]:
    return hopf.star(a, b)


def _scale_levels(levels: list[jax.Array], scalar: float) -> list[jax.Array]:
    return [scalar * term for term in levels]


def _add_levels(a: list[jax.Array], b: list[jax.Array]) -> list[jax.Array]:
    return [x + y for x, y in zip(a, b)]


def _star_power(hopf: HopfAlgebra, x: list[jax.Array], k: int) -> list[jax.Array]:
    """Compute x^{⋆k} with truncation implied by x's depth (k >= 1)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    result = x
    for _ in range(1, k):
        result = hopf.star(result, x)
    return result


def truncated_exp_star(hopf: HopfAlgebra, x: list[jax.Array]) -> list[jax.Array]:
    """exp_{⋆}(x) truncated to the degree of x. Degree 0 term (1) omitted by convention."""
    if len(x) == 0:
        return []
    dtype = x[0].dtype
    depth = len(x)
    acc = hopf.zero(depth, dtype)
    # k = 1 term
    factorial = 1.0
    current_power = x
    acc = _add_levels(acc, _scale_levels(current_power, 1.0 / factorial))
    # k >= 2
    for k in range(2, depth + 1):
        factorial *= float(k)
        current_power = hopf.star(current_power, x)
        acc = _add_levels(acc, _scale_levels(current_power, 1.0 / factorial))
    return acc


def truncated_log_star(hopf: HopfAlgebra, g: list[jax.Array]) -> list[jax.Array]:
    """log_{⋆}(1 + (g-1)) truncated to the degree of g. Degree 0 term omitted by convention.

    Here g is represented without the degree-0 component; algebraically delta = g - 1.
    Series: sum_{k=1..N} (-1)^{k+1} (delta^{⋆k}) / k
    """
    if len(g) == 0:
        return []
    dtype = g[0].dtype
    depth = len(g)
    acc = hopf.zero(depth, dtype)
    # k = 1 term
    current_power = g
    coeff = 1.0
    acc = _add_levels(acc, _scale_levels(current_power, coeff))
    # k >= 2
    for k in range(2, depth + 1):
        current_power = hopf.star(current_power, g)
        coeff = ((-1.0) ** (k + 1)) / float(k)
        acc = _add_levels(acc, _scale_levels(current_power, coeff))
    return acc

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

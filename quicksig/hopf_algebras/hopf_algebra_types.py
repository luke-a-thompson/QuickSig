from typing import NamedTuple, NewType, final, override
from dataclasses import dataclass
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from quicksig.tensor_ops import cauchy_convolution


class HopfAlgebra(ABC):
    """Abstract Hopf algebra interface sufficient for signature/log-signature workflows."""

    ambient_dimension: int

    @abstractmethod
    def basis_size(self, level: int) -> int:
        """Number of basis elements at given level (degree = level + 1 for signatures).

        Implementations must define how many coefficients live at each level.
        """
        raise NotImplementedError

    @abstractmethod
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        """Product on truncated coefficients, degree-wise (omits degree 0).

        Args:
            a_levels: list of flattened tensors per degree (omits degree 0)
            b_levels: same shape/depth as a_levels
        Returns:
            list of flattened tensors per degree representing a ⋆ b
        """
        raise NotImplementedError

    @abstractmethod
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        """Coproduct (deconcatenation) listing splits per degree.

        For degree n (index n-1), return a flat list encoding the pairs
        (deg k, deg n-k) for all splits k=1..n-1. Degree-0 parts are omitted.
        """
        raise NotImplementedError

    @abstractmethod
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        """Exponential with respect to the product, truncated to x's depth (omits degree 0)."""
        raise NotImplementedError

    @abstractmethod
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        """Logarithm with respect to the product, truncated to g's depth (omits degree 0)."""
        raise NotImplementedError

    def zero(self, depth: int, dtype: jnp.dtype) -> list[jax.Array]:
        return [jnp.zeros((self.basis_size(i),), dtype=dtype) for i in range(depth)]

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


@dataclass(frozen=True)
class ShuffleHopfAlgebra(HopfAlgebra):
    """Shuffle/Tensor Hopf algebra used for path signatures.

    The representation uses per-degree flattened tensors, omitting degree 0.
    """

    ambient_dimension: int

    @override
    def basis_size(self, level: int) -> int:
        from quicksig.analytics import get_signature_dim

        return get_signature_dim(level, self.ambient_dimension)

    def _unflatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        dim = self.ambient_dimension
        return [term.reshape((dim,) * (i + 1)) for i, term in enumerate(levels)]

    def _flatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        return [term.reshape(-1) for term in levels]

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncations must match for product.")
        depth = len(a_levels)
        # Work in unflattened tensor shapes for the convolution; then re-flatten
        a_unflat = self._unflatten_levels(a_levels)
        b_unflat = self._unflatten_levels(b_levels)
        cross_unflat = cauchy_convolution(a_unflat, b_unflat, depth)
        out_unflat = [a + b + c for a, b, c in zip(a_unflat, b_unflat, cross_unflat)]
        return self._flatten_levels(out_unflat)

    def _pure_product(
        self, a_levels: list[jax.Array], b_levels: list[jax.Array]
    ) -> list[jax.Array]:
        """(a ⋆ b) with linear parts removed; used by exp/log series."""
        ab = self.product(a_levels, b_levels)
        return [x - y - z for x, y, z in zip(ab, a_levels, b_levels)]

    @override
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        depth = len(levels)
        result: list[list[jax.Array]] = []
        for n in range(1, depth + 1):
            splits: list[jax.Array] = []
            for k in range(1, n):
                splits.append(levels[k - 1])
                splits.append(levels[n - k - 1])
            result.append(splits)
        return result

    @override
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        if len(x) == 0:
            return []
        depth = len(x)
        acc = self.zero(depth, dtype=x[0].dtype)
        factorial = 1.0
        current_power = x  # k = 1
        acc = [a + (1.0 / factorial) * cp for a, cp in zip(acc, current_power)]
        for k in range(2, depth + 1):
            factorial *= float(k)
            current_power = self._pure_product(current_power, x)
            acc = [a + (1.0 / factorial) * cp for a, cp in zip(acc, current_power)]
        return acc

    @override
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        if len(g) == 0:
            return []
        dtype = g[0].dtype
        depth = len(g)
        acc = self.zero(depth, dtype)
        current_power = g  # k = 1
        coeff = 1.0
        acc = [a + coeff * cp for a, cp in zip(acc, current_power)]
        for k in range(2, depth + 1):
            current_power = self._pure_product(current_power, g)
            coeff = ((-1.0) ** (k + 1)) / float(k)
            acc = [a + coeff * cp for a, cp in zip(acc, current_power)]
        return acc

    @override
    def __str__(self) -> str:
        return "Shuffle Hopf Algebra"


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


@dataclass(frozen=True)
class GLHopfAlgebra(HopfAlgebra):
    """Grossman-Larson / Connes-Kreimer Hopf algebra on unordered rooted forests.

    basis_size(level): number of unordered rooted forests with (level+1) nodes,
    multiplied by ambient_dim^(level+1) if nodes are coloured by driver components.
    """

    ambient_dimension: int

    @override
    def basis_size(self, level: int) -> int:
        # level i corresponds to degree n = i+1
        from quicksig.analytics import get_bck_signature_dim

        return get_bck_signature_dim(level, self.ambient_dimension)

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        raise NotImplementedError("GLHopfAlgebra.product is not implemented yet.")

    @override
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        raise NotImplementedError("GLHopfAlgebra.coproduct is not implemented yet.")

    @override
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        raise NotImplementedError("GLHopfAlgebra.exp is not implemented yet.")

    @override
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        raise NotImplementedError("GLHopfAlgebra.log is not implemented yet.")

    @override
    def __str__(self) -> str:
        return "Grossman-Larson Hopf Algebra"


@final
@dataclass(frozen=True)
class MKWHopfAlgebra(HopfAlgebra):
    """Munthe-Kaas-Wright Hopf algebra on ordered (planar) rooted forests.

    basis_size(level): number of plane rooted forests with (level+1) nodes,
    multiplied by ambient_dim^(level+1) if nodes are coloured by driver components.
    """

    ambient_dimension: int

    @override
    def basis_size(self, level: int) -> int:
        from quicksig.analytics import get_mkw_signature_dim

        return get_mkw_signature_dim(level, self.ambient_dimension)

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        raise NotImplementedError("MKWHopfAlgebra.product is not implemented yet.")

    @override
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        raise NotImplementedError("MKWHopfAlgebra.coproduct is not implemented yet.")

    @override
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        raise NotImplementedError("MKWHopfAlgebra.exp is not implemented yet.")

    @override
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        raise NotImplementedError("MKWHopfAlgebra.log is not implemented yet.")

    @override
    def __str__(self) -> str:
        return "Munthe-Kaas-Wright Hopf Algebra"

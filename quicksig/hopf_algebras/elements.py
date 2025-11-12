from __future__ import annotations

from dataclasses import dataclass
from typing import final, override, TypeVar
import jax
import jax.numpy as jnp

from quicksig.hopf_algebras.hopf_algebra_types import HopfAlgebra

TGroup = TypeVar("TGroup", bound="GroupElement")


@final
@dataclass(frozen=True)
class GroupElement:
    """Group-like element in H^g, represented by its tail (degrees >= 1)."""

    hopf: HopfAlgebra
    coeffs: list[jax.Array]
    interval: tuple[float, float]

    def star(self: TGroup, other: TGroup) -> TGroup:
        if self.hopf != other.hopf:
            raise ValueError("Cannot multiply elements from different Hopf algebras.")
        if self.depth != other.depth:
            raise ValueError("Cannot multiply elements with different depths.")
        if self.ambient_dimension != other.ambient_dimension:
            raise ValueError("Cannot multiply elements with different ambient dimensions.")
        jax.debug.callback(
            lambda b: None
            if not bool(b)
            else (_ for _ in ()).throw(
                ValueError("Cannot multiply elements with non-consecutive intervals.")
            ),
            jnp.asarray(other.interval[0]) != jnp.asarray(self.interval[1]),
        )
        new_coeffs = self.hopf.product(self.coeffs, other.coeffs)
        return type(self)(self.hopf, new_coeffs, (self.interval[0], other.interval[1]))

    def __matmul__(self: TGroup, other: TGroup) -> TGroup:
        return self.star(other)

    def tree_flatten(
        self,
    ) -> tuple[tuple[list[jax.Array]], tuple[HopfAlgebra, tuple[float, float]]]:
        children = (self.coeffs,)
        aux = (self.hopf, self.interval)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[HopfAlgebra, tuple[float, float]],
        children: tuple[list[jax.Array]],
    ) -> GroupElement:
        hopf, interval = aux
        (coeffs,) = children
        return cls(hopf=hopf, coeffs=coeffs, interval=interval)

    def log(self) -> LieElement:
        coeffs = self.hopf.log(self.coeffs)
        return LieElement(self.hopf, coeffs, self.interval)

    def flatten(self) -> jax.Array:
        if len(self.coeffs) == 0:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate([jnp.ravel(term) for term in self.coeffs], axis=0)

    @property
    def depth(self) -> int:
        return len(self.coeffs)

    @property
    def ambient_dimension(self) -> int:
        return int(self.hopf.ambient_dimension)

    @override
    def __str__(self) -> str:
        information = f"""
        An element of the {self.hopf} Hopf algebra.
        Depth: {self.depth}
        Ambient dimension: {self.ambient_dimension}
        Interval: {self.interval}
        """
        return information


@final
@dataclass(frozen=True)
class LieElement:
    """Lie-like element in P(H^g), represented by its tail (degrees >= 1)."""

    hopf: HopfAlgebra
    coeffs: list[jax.Array]
    interval: tuple[float, float]

    def exp(self) -> GroupElement:
        coeffs = self.hopf.exp(self.coeffs)
        return GroupElement(self.hopf, coeffs, self.interval)

    # JAX pytree support -------------------------------------------------------
    def tree_flatten(
        self,
    ) -> tuple[tuple[list[jax.Array]], tuple[HopfAlgebra, tuple[float, float]]]:
        children = (self.coeffs,)
        aux = (self.hopf, self.interval)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[HopfAlgebra, tuple[float, float]],
        children: tuple[list[jax.Array]],
    ) -> LieElement:
        hopf, interval = aux
        (coeffs,) = children
        return cls(hopf=hopf, coeffs=coeffs, interval=interval)

    def flatten(self) -> jax.Array:
        if len(self.coeffs) == 0:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate([jnp.ravel(term) for term in self.coeffs], axis=0)

    @property
    def depth(self) -> int:
        return len(self.coeffs)

    @property
    def ambient_dimension(self) -> int:
        return int(self.hopf.ambient_dimension)


_ = jax.tree_util.register_pytree_node_class(GroupElement)
_ = jax.tree_util.register_pytree_node_class(LieElement)

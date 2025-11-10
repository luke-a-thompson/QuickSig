from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from quicksig.hopf_algebras.hopf_algebra_types import HopfAlgebra, convolution_star, truncated_exp_star, truncated_log_star


@dataclass(frozen=True)
class GroupElement:
    hopf: HopfAlgebra
    coeffs: list[jax.Array]
    interval: tuple[float, float]

    def star(self, other: "GroupElement") -> "GroupElement":
        if type(self.hopf) is not type(other.hopf):
            raise ValueError("Cannot multiply elements from different Hopf algebras.")
        new_coeffs = convolution_star(self.hopf, self.coeffs, other.coeffs)
        return GroupElement(self.hopf, new_coeffs, (self.interval[0], other.interval[1]))

    def __matmul__(self, other: "GroupElement") -> "GroupElement":
        return self.star(other)

    def log(self) -> "LieElement":
        coeffs = truncated_log_star(self.hopf, self.coeffs)
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
        return int(self.hopf.ambient_dimension())


@dataclass(frozen=True)
class LieElement:
    hopf: HopfAlgebra
    coeffs: list[jax.Array]
    interval: tuple[float, float]

    def exp(self) -> GroupElement:
        coeffs = truncated_exp_star(self.hopf, self.coeffs)
        return GroupElement(self.hopf, coeffs, self.interval)

    def flatten(self) -> jax.Array:
        if len(self.coeffs) == 0:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate([jnp.ravel(term) for term in self.coeffs], axis=0)

    @property
    def depth(self) -> int:
        return len(self.coeffs)

    @property
    def ambient_dimension(self) -> int:
        return int(self.hopf.ambient_dimension())


_ = jax.tree_util.register_pytree_node_class(GroupElement)
_ = jax.tree_util.register_pytree_node_class(LieElement)



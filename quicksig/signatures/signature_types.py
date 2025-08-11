from dataclasses import dataclass
from typing import Literal, override
from abc import ABC
import jax
import jax.numpy as jnp
from quicksig.tensor_ops import cauchy_convolution


@dataclass(frozen=True)
class BaseSignature(ABC):
    signature: list[jax.Array]
    interval: tuple[float, float]
    ambient_dimension: int
    depth: int
    basis_name: Literal["Tensor words", "Lyndon words"]

    def flatten(self) -> jax.Array:
        """Flattens the signature terms into a single vector."""
        return jnp.concatenate([jnp.ravel(term) for term in self.signature], axis=0)

    @override
    def __str__(self) -> str:
        string = f"""{self.__class__.__name__}(
    depth={self.depth},
    ambient_dimension={self.ambient_dimension},
    interval={self.interval},
    signature_shapes={[term.shape for term in self.signature]}
)"""
        return string

    def tree_flatten(self):
        """Flattens the Pytree."""
        children = (self.signature,)
        aux_data = {
            "interval": self.interval,
            "ambient_dimension": self.ambient_dimension,
            "depth": self.depth,
            "basis_name": self.basis_name,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflattens the Pytree."""
        (signature,) = children
        return cls(signature=signature, **aux_data)


@dataclass(frozen=True)
class Signature(BaseSignature):
    basis_name: Literal["Tensor words"] = "Tensor words"

    def __matmul__(self, other: "Signature") -> "Signature":
        return _chen_identity(self, other)


@dataclass(frozen=True)
class LogSignature(BaseSignature):
    basis_name: Literal["Tensor words", "Lyndon words"]

    def __matmul__(self, other: "LogSignature") -> "LogSignature":
        raise NotImplementedError("Product of log signatures is not defined.")


jax.tree_util.register_pytree_node_class(Signature)
jax.tree_util.register_pytree_node_class(LogSignature)


def _chen_identity(lhs_signature: Signature, rhs_signature: Signature) -> Signature:
    if lhs_signature.ambient_dimension != rhs_signature.ambient_dimension:
        raise ValueError("Signatures must have the same ambient_dimension.")
    if lhs_signature.depth != rhs_signature.depth:
        raise ValueError("Signatures must have the same depth. " f"Got: {lhs_signature.depth} and {rhs_signature.depth}")

    if lhs_signature.interval[1] != rhs_signature.interval[0]:
        gap = rhs_signature.interval[0] - lhs_signature.interval[1]
        raise ValueError(
            f"""The intervals of the signatures must be contiguous.
Gap of size {gap} found between lhs signature (interval {lhs_signature.interval})
and rhs signature (interval {rhs_signature.interval})."""
        )

    ambient_dim = lhs_signature.ambient_dimension
    depth = lhs_signature.depth

    # Reshape flattened signature terms back to their tensor structure
    lhs_unflat = [term.reshape((ambient_dim,) * (i + 1)) for i, term in enumerate(lhs_signature.signature)]
    rhs_unflat = [term.reshape((ambient_dim,) * (i + 1)) for i, term in enumerate(rhs_signature.signature)]

    cross_terms_unflat = cauchy_convolution(lhs_unflat, rhs_unflat, depth, lhs_unflat)

    new_signature_terms_unflat = [lhs + rhs + cross for lhs, rhs, cross in zip(lhs_unflat, rhs_unflat, cross_terms_unflat)]

    # Flatten the terms back to the convention used by compute_path_signature
    new_signature_terms_flat = [term.flatten() for term in new_signature_terms_unflat]

    return Signature(
        signature=new_signature_terms_flat,
        interval=(lhs_signature.interval[0], rhs_signature.interval[1]),
        ambient_dimension=lhs_signature.ambient_dimension,
        depth=lhs_signature.depth,
    )

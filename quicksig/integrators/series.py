import jax
import jax.numpy as jnp
from quicksig.integrators.integrator_types import ButcherSeries, LieSeries, LieButcherSeries
from quicksig.vector_field_lifts.vector_field_lift_types import (
    ButcherDifferentials,
    LieButcherDifferentials,
)


def form_butcher_series(
    differentials: ButcherDifferentials | LieButcherDifferentials,
    coefficients: list[jax.Array],
) -> ButcherSeries | LieButcherSeries:
    coefficients_flat: jax.Array = (
        jnp.concatenate(coefficients, axis=0) if coefficients else jnp.zeros((0,))
    )

    if differentials.shape[0] != coefficients_flat.shape[0]:
        raise ValueError(
            f"Coefficient count {coefficients_flat.shape[0]} does not match number of basis terms {differentials.shape[0]}."
        )

    series = jnp.tensordot(coefficients_flat, differentials, axes=1)

    if isinstance(differentials, ButcherDifferentials):
        return ButcherSeries(series)
    elif isinstance(differentials, LieButcherDifferentials):
        return LieButcherSeries(series)


def form_lie_series(
    basis_terms: jax.Array,
    lam_by_len: list[jax.Array],
    words_by_len: list[jax.Array],
) -> LieSeries:
    """Form the Lie series matrix: C = sum_w lam_w * W[w]."""
    lams = [lam for lam, words in zip(lam_by_len, words_by_len) if words.size != 0]
    coefficients_flat: jax.Array = jnp.concatenate(lams, axis=0) if lams else jnp.zeros((0,))

    if basis_terms.shape[0] != coefficients_flat.shape[0]:
        raise ValueError(
            f"Coefficient count {coefficients_flat.shape[0]} does not match number of basis terms {basis_terms.shape[0]}."
        )
    series = jnp.tensordot(coefficients_flat, basis_terms, axes=1)
    return LieSeries(series)

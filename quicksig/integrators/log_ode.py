import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm
from quicksig.integrators.series import form_lie_series
from quicksig.vector_field_lifts.vector_field_lift_types import LyndonBrackets


@jax.jit
def log_ode(
    vector_field_brackets: LyndonBrackets,
    lyndon_signature: list[jax.Array],
    words_by_len: list[jax.Array],
    curr_state: jax.Array,
) -> jax.Array:
    polynomial = form_lie_series(vector_field_brackets, lyndon_signature, words_by_len)
    exp_polynomial: jax.Array = jexpm(polynomial)
    return (exp_polynomial @ curr_state) / jnp.linalg.norm(curr_state)

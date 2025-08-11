import itertools
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import jacrev
from jax.experimental.ode import odeint
from fbm import fbm

from quicksig.signatures import compute_log_signature
from quicksig.log_signature import duval_generator


def V1(y: jnp.ndarray) -> jnp.ndarray:
    """
    Complex nonlinear vector field: V1(y) = [y[1], -y[0]]
    Represents a rotation by 90° in phase space.
    """
    return jnp.array([y[1], -y[0]])


def V2(y: jnp.ndarray) -> jnp.ndarray:
    """
    Complex nonlinear vector field: V2(y) = [sin(y[0]), cos(y[1])]
    Introduces nonlinearity via sine and cosine.
    """
    return jnp.array([jnp.sin(y[0]), jnp.cos(y[1])])


def lie_bracket(V: Callable, W: Callable) -> Callable:
    """Return a callable y ↦ [V,W](y) using true matrix-vector products."""
    J_V = jacrev(V)
    J_W = jacrev(W)

    def bracket(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(J_W(y), V(y)) - jnp.matmul(J_V(y), W(y))

    return bracket


def build_brackets(base_fields: Sequence[Callable], depth: int) -> dict[tuple[int, ...], Callable]:
    """
    Build all iterated Lie brackets V_I up to given depth.
    """
    B: dict[tuple[int, ...], Callable] = {(i,): f for i, f in enumerate(base_fields)}
    for m in range(2, depth + 1):
        for I in itertools.product(range(len(base_fields)), repeat=m):
            i1, *rest = I
            B[I] = lie_bracket(base_fields[i1], B[tuple(rest)])
    return B


def log_ode_step(
    y0: jnp.ndarray,
    L_coeffs: jnp.ndarray,
    brackets: dict[tuple[int, ...], Callable],
    depth: int,
) -> jnp.ndarray:
    """
    Perform one log-ODE step: integrate ẋ = Σ_I L_I V_I(z) from τ=0 to 1.
    """
    driver_dim = len([I for I in brackets if len(I) == 1])
    lyndon_lists = duval_generator(depth, driver_dim)
    lyndon_words = []
    for level in lyndon_lists:
        for row in level:  # type: ignore
            lyndon_words.append(tuple(row.tolist()))
    bracket_fns = [brackets[w] for w in lyndon_words]

    def rhs(z: jnp.ndarray, _tau: float) -> jnp.ndarray:
        out = jnp.zeros_like(z)
        for coeff, V_I in zip(L_coeffs, bracket_fns):
            out = out + coeff * V_I(z)
        return out

    z_T = odeint(rhs, y0, jnp.array([0.0, 1.0]))[-1]
    return z_T


def solve_rde_logode(
    y0: jnp.ndarray,
    X: jnp.ndarray,
    blocksize: int,
    depth: int,
    brackets: dict[tuple[int, ...], Callable],
    key: jax.Array,
) -> jnp.ndarray:
    """
    Solve RDE via sequential log-ODE steps over each two-point segment.
    """
    num_segments = X.shape[0] - 1
    if num_segments % blocksize != 0:
        raise ValueError("Number of segments must be divisible by blocksize")
    y = y0
    for k in range(0, num_segments, blocksize):
        segment = X[k : k + blocksize + 1]
        L = get_log_signature(segment, depth, "lyndon")
        y = log_ode_step(y, L, brackets, depth)
    return y


if __name__ == "__main__":
    import time

    key = jax.random.PRNGKey(0)
    base_fields = (V1, V2)
    depth = 2
    brackets = build_brackets(base_fields, depth)

    # Rough FBM driver
    num_points = 101
    num_segments = num_points - 1
    blocksize = 10
    H = 0.33
    t_grid = jnp.linspace(0.0, 1.0, num_points)
    X1 = fbm(n=num_segments, hurst=H, length=1, method="daviesharte")
    X2 = fbm(n=num_segments, hurst=H, length=1, method="daviesharte")
    X = jnp.stack([X1, X2], axis=1)

    y0 = jnp.array([1.0, 0.0])

    # log-ODE solve
    time_start = time.time()
    y_logode = solve_rde_logode(y0, X, blocksize, depth, brackets, key)
    print(f"Log-ODE solve time: {time.time() - time_start}")
    print("Log-ODE final state:", y_logode)

    dX = num_segments * (X[1:] - X[:-1])  # shape (num_segments, 2)

    def standard_rhs(y: jnp.ndarray, t: float) -> jnp.ndarray:
        kf = t * num_segments
        ki = jnp.clip(jnp.floor(kf).astype(jnp.int32), 0, num_segments - 1)
        dx = jnp.take(dX, ki, axis=0)
        return V1(y) * dx[0] + V2(y) * dx[1]

    time_start = time.time()
    y_standard = odeint(standard_rhs, y0, t_grid)[-1]
    print(f"Standard ODE solve time: {time.time() - time_start}")

    print("Standard ODE final state:", y_standard)
    print("Difference norm:", jnp.linalg.norm(y_logode - y_standard))

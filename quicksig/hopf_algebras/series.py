import jax
import jax.numpy as jnp


def lie_polynomial(bracket_basis: jax.Array, lie_coefficients: jax.Array) -> jax.Array:
    """Form the Lie series matrix: C = sum_w lam_w * W[w].

    A lie polynomial is a linear combination of Lie brackets (commutators).
    It is a truncation of an infinite Lie series.

    Works with any bracket basis (e.g., right-normed or Lyndon) as long as the
    coefficients are in the same ordering as the brackets.

    Args:
        bracket_basis: [L, n, n] array of Lie brackets (e.g., from form_lyndon_brackets).
        lie_coefficients: [L] array of coefficients (e.g., flattened log signature in Lyndon coordinates).

    Returns:
        [n, n] array of the Lie series matrix.

    Example:
        For log signature in Lyndon coordinates:
        >>> log_sig = compute_log_signature(path, depth, log_signature_type="Lyndon words")
        >>> brackets = form_lyndon_brackets(A, depth, dim)
        >>> coeffs = flatten_coeffs(log_sig.signature, duval_generator(depth, dim))
        >>> lie_elem = lie_polynomial(brackets, coeffs)
    """
    if bracket_basis.shape[0] != lie_coefficients.shape[0]:
        raise ValueError(
            f"Coefficient count {lie_coefficients.shape[0]} does not match number of brackets {bracket_basis.shape[0]}."
        )
    return jnp.tensordot(lie_coefficients, bracket_basis, axes=1)  # [n, n]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jax.scipy.linalg import expm as jexpm

    from quicksig.signatures import compute_log_signature
    from quicksig.signatures.compute_log_signature import duval_generator
    from quicksig.hopf_algebras.free_lie import (
        form_lyndon_brackets,
        flatten_coeffs,
    )
    from quicksig.hopf_algebras.series import lie_polynomial

    # Brownian motion in R^3 driving rotations on S^2 via SO(3) action
    key = jax.random.PRNGKey(0)
    N: int = 1000
    dim: int = 3
    depth: int = 3  # include commutators
    dt: float = 1.0 / float(N)
    dW: jax.Array = jax.random.normal(key, shape=(N, dim)) * jnp.sqrt(dt)
    W: jax.Array = jnp.vstack([jnp.zeros((1, dim)), jnp.cumsum(dW, axis=0)])
    # Second, independent Brownian path
    key2 = jax.random.PRNGKey(1)
    dW2: jax.Array = jax.random.normal(key2, shape=(N, dim)) * jnp.sqrt(dt)
    W2: jax.Array = jnp.vstack([jnp.zeros((1, dim)), jnp.cumsum(dW2, axis=0)])

    # so(3) generators (standard basis)
    def so3_generators() -> jax.Array:
        A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        return jnp.stack([A1, A2, A3], axis=0)

    A: jax.Array = so3_generators()
    words_by_len: list[jax.Array] = duval_generator(depth, dim)
    bracket_basis: jax.Array = form_lyndon_brackets(A, depth, dim)

    # initial point on the sphere S^2
    y0: jax.Array = jnp.array([0.0, 0.0, 1.0])
    y: jax.Array = y0
    traj: list[jax.Array] = [y0]
    # Second trajectory
    y2: jax.Array = y0
    traj2: list[jax.Array] = [y0]

    window: int = 10  # windowed log-signatures to create non-trivial commutators
    for s in range(0, N, window):
        e = min(s + window, N)
        segment: jax.Array = W[s : e + 1, :]
        log_sig = compute_log_signature(segment, depth, "Lyndon words", mode="full")
        lam: jax.Array = flatten_coeffs(log_sig.signature, words_by_len)
        C: jax.Array = lie_polynomial(bracket_basis, lam)  # [3, 3]
        y = jexpm(C) @ y
        y = y / jnp.linalg.norm(y)
        traj.append(y)

    # Evolve second path
    for s in range(0, N, window):
        e = min(s + window, N)
        segment2: jax.Array = W2[s : e + 1, :]
        log_sig2 = compute_log_signature(segment2, depth, "Lyndon words", mode="full")
        lam2: jax.Array = flatten_coeffs(log_sig2.signature, words_by_len)
        C2: jax.Array = lie_polynomial(bracket_basis, lam2)  # [3, 3]
        y2 = jexpm(C2) @ y2
        y2 = y2 / jnp.linalg.norm(y2)
        traj2.append(y2)

    traj_arr: jax.Array = jnp.stack(traj, axis=0)
    traj2_arr: jax.Array = jnp.stack(traj2, axis=0)

    # Render on a sphere
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    u = jnp.linspace(0.0, 2.0 * jnp.pi, 60)
    v = jnp.linspace(0.0, jnp.pi, 30)
    X = jnp.outer(jnp.cos(u), jnp.sin(v))
    Y = jnp.outer(jnp.sin(u), jnp.sin(v))
    Z = jnp.outer(jnp.ones_like(u), jnp.cos(v))
    ax.plot_surface(X, Y, Z, color="lightgray", alpha=0.25, linewidth=0)

    ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], color="C3", lw=2)
    ax.plot(traj2_arr[:, 0], traj2_arr[:, 1], traj2_arr[:, 2], color="C0", lw=2)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title("Log-ODE on S^2 via Lyndon log-signature")
    plt.tight_layout()
    plt.show()

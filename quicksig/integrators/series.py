import jax
import jax.numpy as jnp


def form_series(
    basis_terms: jax.Array,
    lam_by_len: list[jax.Array],
    words_by_len: list[jax.Array],
) -> jax.Array:
    """Form the Lie series matrix: C = sum_w lam_w * W[w]."""
    lams = [lam for lam, words in zip(lam_by_len, words_by_len) if words.size != 0]
    coefficients_flat: jax.Array = (
        jnp.concatenate(lams, axis=0) if lams else jnp.zeros((0,), dtype=jnp.float32)
    )

    if basis_terms.shape[0] != coefficients_flat.shape[0]:
        raise ValueError(
            f"Coefficient count {coefficients_flat.shape[0]} does not match number of basis terms {basis_terms.shape[0]}."
        )
    return jnp.tensordot(coefficients_flat, basis_terms, axes=1)  # [n, n]


def log_ode(
    vector_field_lift: jax.Array,
    lie_coefficients_by_len: list[jax.Array],
    words_by_len: list[jax.Array],
    y: jax.Array,
) -> jax.Array:
    polynomial: jax.Array = form_series(vector_field_lift, lie_coefficients_by_len, words_by_len)
    exp_polynomial: jax.Array = jexpm(polynomial)
    return (exp_polynomial @ y) / jnp.linalg.norm(y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jax.scipy.linalg import expm as jexpm

    from quicksig.control_lifts import compute_log_signature
    from quicksig.control_lifts.log_signature import duval_generator
    from quicksig.hopf_algebras.free_lie import form_lyndon_brackets
    from quicksig.integrators.series import form_series

    key = jax.random.PRNGKey(0)
    N: int = 1000
    dim: int = 3
    depth: int = 3
    dt: float = 1.0 / float(N)
    dW: jax.Array = jax.random.normal(key, shape=(N, dim)) * jnp.sqrt(dt)
    W: jax.Array = jnp.vstack([jnp.zeros((1, dim)), jnp.cumsum(dW, axis=0)])
    key2 = jax.random.PRNGKey(1)
    dW2: jax.Array = jax.random.normal(key2, shape=(N, dim)) * jnp.sqrt(dt)
    W2: jax.Array = jnp.vstack([jnp.zeros((1, dim)), jnp.cumsum(dW2, axis=0)])

    def so3_generators() -> jax.Array:
        A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        return jnp.stack([A1, A2, A3], axis=0)

    A: jax.Array = so3_generators()
    words_by_len: list[jax.Array] = duval_generator(depth, dim)
    bracket_basis: jax.Array = form_lyndon_brackets(A, depth, dim)

    y0: jax.Array = jnp.array([0.0, 0.0, 1.0])
    y: jax.Array = y0
    traj: list[jax.Array] = [y0]
    y2: jax.Array = y0
    traj2: list[jax.Array] = [y0]

    window: int = 10
    for s in range(0, N, window):
        e = min(s + window, N)
        segment: jax.Array = W[s : e + 1, :]
        log_sig = compute_log_signature(segment, depth, "Lyndon words", mode="full")
        C: jax.Array = form_series(bracket_basis, log_sig.signature, words_by_len)  # [3, 3]
        y = jexpm(C) @ y
        y = y / jnp.linalg.norm(y)
        traj.append(y)

    for s in range(0, N, window):
        e = min(s + window, N)
        segment2: jax.Array = W2[s : e + 1, :]
        log_sig2 = compute_log_signature(segment2, depth, "Lyndon words", mode="full")
        C2: jax.Array = form_series(bracket_basis, log_sig2.signature, words_by_len)  # [3, 3]
        y2 = jexpm(C2) @ y2
        y2 = y2 / jnp.linalg.norm(y2)
        traj2.append(y2)

    traj_arr: jax.Array = jnp.stack(traj, axis=0)
    traj2_arr: jax.Array = jnp.stack(traj2, axis=0)

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

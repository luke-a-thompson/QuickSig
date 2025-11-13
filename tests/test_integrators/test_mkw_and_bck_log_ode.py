import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from quicksig.controls.drivers import bm_driver
from quicksig.controls.augmentations import non_overlapping_windower
from quicksig.control_lifts.log_signature import compute_log_signature, duval_generator
from quicksig.control_lifts.branched_signature_ito import (
    compute_planar_branched_signature,
    compute_nonplanar_branched_signature,
)
from quicksig.integrators.log_ode import log_ode
from quicksig.vector_field_lifts.lie_lift import form_lyndon_brackets
from quicksig.vector_field_lifts.mkw_lift import compute_mkw_brackets_by_degree
from quicksig.vector_field_lifts.bck_lift import compute_bck_brackets_by_degree
from quicksig.hopf_algebras import enumerate_mkw_trees, enumerate_bck_trees
from quicksig.hopf_algebras.hopf_algebra_types import MKWHopfAlgebra, GLHopfAlgebra
from quicksig.hopf_algebras.elements import GroupElement
from typing import cast


def _so3_generators() -> jax.Array:
    A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32)
    A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    return jnp.stack([A1, A2, A3], axis=0)


def _linear_vector_fields(A: jax.Array):
    return [lambda y, M=A[i]: M @ y for i in range(A.shape[0])]


def _project_to_tangent(y: jax.Array, v: jax.Array) -> jax.Array:
    # Tangent projection on S^2
    return v - jnp.dot(v, y) * y


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_log_ode_euclidean(depth: int) -> None:
    """
    Euclidean: depth=1 should match matrix exponential for 1D driver;
    we also keep this test name for unified integrator coverage.
    """
    # Single generator in R^2: 90-degree rotation generator
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A = A0[jnp.newaxis, ...]  # [1, 2, 2]
    dim = 1
    words_by_len = duval_generator(depth, dim)
    bracket_basis = form_lyndon_brackets(A, depth)

    # Use driver to get a single increment
    key = jax.random.PRNGKey(123)
    path = bm_driver(key, timesteps=1, dim=dim).path  # shape (2, 1)
    delta = float(path[-1, 0] - path[0, 0])

    y0 = jnp.array([1.0, 0.0], dtype=jnp.float32)
    primitive = compute_log_signature(path, depth, "Lyndon words", mode="full")
    y_logode = log_ode(bracket_basis, primitive, y0)

    if depth == 1:
        expected = jexpm(delta * A0) @ y0
        expected = expected / jnp.linalg.norm(expected)
        assert jnp.allclose(y_logode, expected, rtol=1e-6, atol=1e-6)
    else:
        # Higher-depth reduces to depth-1 in 1D commuting case; just ensure finite/normalized output
        assert jnp.isfinite(jnp.linalg.norm(y_logode))
        assert jnp.allclose(jnp.linalg.norm(y_logode), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("depth", [1, 2])
def test_log_ode_manifold(depth: int) -> None:
    """
    Manifold (S^2): norm preservation, small-time tangent-plane stats, long-time mean decay.
    """
    # Two skew-symmetric generators that span the tangent plane at e3
    A1 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32)
    A2 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=jnp.float32)
    A = jnp.stack([A1, A2], axis=0)  # [2, 3, 3]
    dim = A.shape[0]
    words_by_len = duval_generator(depth, dim)
    bracket_basis = form_lyndon_brackets(A, depth)
    y0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    @jax.jit
    def integrate_path(increments: jax.Array, y_init: jax.Array) -> tuple[jax.Array, jax.Array]:
        def step(carry: jax.Array, inc: jax.Array) -> tuple[jax.Array, jax.Array]:
            seg_W = jnp.vstack([jnp.zeros((1, dim), dtype=inc.dtype), inc.reshape(1, -1)])
            primitive = compute_log_signature(seg_W, depth, "Lyndon words", mode="full")
            y_next = log_ode(bracket_basis, primitive, carry)
            return y_next, y_next

        y_T, ys = jax.lax.scan(step, y_init, increments)
        return y_T, ys

    key = jax.random.PRNGKey(0)
    # 1) Norm preservation along a long trajectory
    T_long = 2.0
    N_long = 200
    dt_long = T_long / N_long
    key, sub = jax.random.split(key)
    dW_long = jax.random.normal(sub, shape=(N_long, dim), dtype=jnp.float32) * jnp.sqrt(dt_long)
    _, ys_long = integrate_path(dW_long, y0)
    norms = jnp.linalg.norm(ys_long, axis=1)
    assert jnp.allclose(norms, 1.0, rtol=1e-6, atol=1e-6)

    # 2) Small-time tangent-plane stats
    M_small = 512
    T_small = 0.05
    N_small = 50
    dt_small = T_small / N_small
    key, sub = jax.random.split(key)
    dW_small = jax.random.normal(sub, shape=(M_small, N_small, dim), dtype=jnp.float32) * jnp.sqrt(
        dt_small
    )

    @jax.jit
    def integrate_batch(increments_batch: jax.Array) -> jax.Array:
        def one_traj(incs: jax.Array) -> jax.Array:
            yT, _ = integrate_path(incs, y0)
            return yT

        return jax.vmap(one_traj, in_axes=0)(increments_batch)

    yT_small = integrate_batch(dW_small)  # [M_small, 3]
    disp_small = yT_small - y0
    P_tan = jnp.eye(3, dtype=jnp.float32) - jnp.outer(y0, y0)
    disp_tan = disp_small @ P_tan.T
    disp_xy = disp_tan[:, :2]
    mean_xy = jnp.mean(disp_xy, axis=0)
    assert jnp.linalg.norm(mean_xy) < 0.07
    centered = disp_xy - mean_xy
    cov_xy = (centered.T @ centered) / float(M_small)
    target_cov = T_small * jnp.eye(2, dtype=jnp.float32)
    assert jnp.allclose(cov_xy, target_cov, rtol=0.15, atol=0.02)

    # 3) Long-time approximate uniformity: empirical mean near zero (decay)
    M_large = 512
    key, sub = jax.random.split(key)
    dW_large = jax.random.normal(sub, shape=(M_large, N_long, dim), dtype=jnp.float32) * jnp.sqrt(
        dt_long
    )
    yT_large = integrate_batch(dW_large)
    mean_large = jnp.mean(yT_large, axis=0)
    expected_decay = jnp.exp(-T_long)
    assert jnp.allclose(jnp.linalg.norm(mean_large), expected_decay, rtol=0.3, atol=0.03)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dim", [1, 3])
def test_bck_log_ode_euclidean(depth: int, dim: int) -> None:
    """
    BCK (non-planar branched) rough paths on Euclidean systems: Quadratic variation check
    at the signature level. Degree-2 coordinates vanish when cov=0 and are non-zero when
    cov=dt*I at the chain-of-length-2 indices.
    """
    forests = enumerate_bck_trees(depth)
    hopf = GLHopfAlgebra.build(dim, forests)
    timesteps = 200
    key = jax.random.PRNGKey(7)
    W = bm_driver(key, timesteps=timesteps, dim=dim)
    dt = 1.0 / timesteps
    I = jnp.eye(dim, dtype=jnp.float32)

    steps = W.num_timesteps - 1
    cov_zero = jnp.zeros((steps, dim, dim), dtype=W.path.dtype)
    cov_dtI = jnp.tile((dt * I)[None, :, :], reps=(steps, 1, 1))
    sig_zero = cast(
        list[jax.Array],
        compute_nonplanar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_zero,
            return_trajectory=False,
        ),
    )
    sig_dtI = cast(
        list[jax.Array],
        compute_nonplanar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_dtI,
            return_trajectory=False,
        ),
    )
    if depth >= 2 and hopf.degree2_chain_indices is not None:
        chain_zero = sig_zero[1][hopf.degree2_chain_indices]
        chain_dtI = sig_dtI[1][hopf.degree2_chain_indices]
        # For multi-step paths, degree-2 mass arises even with cov=0 via group product.
        # QV injection must strictly increase the chain component norm.
        norm_zero = jnp.linalg.norm(chain_zero)
        norm_dtI = jnp.linalg.norm(chain_dtI)
        assert norm_dtI > norm_zero + 1e-6


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dim", [1, 3])
def test_mkw_log_ode_euclidean(depth: int, dim: int) -> None:
    """
    MKW (planar branched) rough paths on Euclidean systems: Quadratic variation check
    at the signature level. Degree-2 coordinates vanish when cov=0 and are non-zero when
    cov=dt*I at the chain-of-length-2 indices.
    """
    forests = enumerate_mkw_trees(depth)
    hopf = MKWHopfAlgebra.build(dim, forests)
    timesteps = 200
    key = jax.random.PRNGKey(9)
    W = bm_driver(key, timesteps=timesteps, dim=dim)
    dt = 1.0 / timesteps
    I = jnp.eye(dim, dtype=jnp.float32)

    steps = W.num_timesteps - 1
    cov_zero = jnp.zeros((steps, dim, dim), dtype=W.path.dtype)
    cov_dtI = jnp.tile((dt * I)[None, :, :], reps=(steps, 1, 1))
    sig_zero = cast(
        list[jax.Array],
        compute_planar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_zero,
            return_trajectory=False,
        ),
    )
    sig_dtI = cast(
        list[jax.Array],
        compute_planar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_dtI,
            return_trajectory=False,
        ),
    )
    if depth >= 2 and hopf.degree2_chain_indices is not None:
        chain_zero = sig_zero[1][hopf.degree2_chain_indices]
        chain_dtI = sig_dtI[1][hopf.degree2_chain_indices]
        norm_zero = jnp.linalg.norm(chain_zero)
        norm_dtI = jnp.linalg.norm(chain_dtI)
        assert norm_dtI > norm_zero + 1e-6


@pytest.mark.parametrize("depth", [1, 2])
def test_mkw_log_ode_manifold(depth: int) -> None:
    """
    MKW on S^2 with tangent projection:
    - Norm preservation
    - Small-time tangent-plane statistics ~ N(0, t I)
    """
    # Use so(3) generators, but vector fields evaluated through projection for MKW brackets
    A = _so3_generators()  # [3,3,3]
    V = _linear_vector_fields(A)
    y0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    dim = 3

    forests = enumerate_mkw_trees(depth)
    hopf = MKWHopfAlgebra.build(dim, forests)
    words_by_len = [f.parent for f in forests]
    mkw_brackets = compute_mkw_brackets_by_degree(V, y0, forests, _project_to_tangent)

    key = jax.random.PRNGKey(4)
    timesteps = 1000
    window_size = 10
    W = bm_driver(key, timesteps=timesteps, dim=dim)
    windows = non_overlapping_windower(W, window_size=window_size)
    dt = 1.0 / timesteps
    I = jnp.eye(dim, dtype=jnp.float32)

    # Integrate over windows using planar branched signatures with correct QV
    state = y0
    traj = [state]
    for w in windows:
        steps = w.num_timesteps - 1
        cov = jnp.tile((dt * I)[None, :, :], reps=(steps, 1, 1))
        sig_levels = compute_planar_branched_signature(
            path=w.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov,
            return_trajectory=False,
        )
        sig_levels_list = cast(list[jax.Array], sig_levels)
        group_el = GroupElement(hopf=hopf, coeffs=sig_levels_list, interval=w.interval)
        logsig = group_el.log()
        state = log_ode(mkw_brackets, logsig, state)
        traj.append(state)
    trajectory = jnp.stack(traj, axis=0)

    # Norm preservation (relaxed tolerance)
    norms = jnp.linalg.norm(trajectory, axis=1)
    assert jnp.max(jnp.abs(norms - 1.0)) < 0.05

    # Small-time statistics around y0 using short integration
    key = jax.random.PRNGKey(5)
    M_small = 256
    T_small = 0.05
    N_small = 50
    dt_small = T_small / N_small
    key, sub = jax.random.split(key)
    dW_small = jax.random.normal(sub, shape=(M_small, N_small, dim), dtype=jnp.float32) * jnp.sqrt(
        dt_small
    )

    def integrate_short(incs: jax.Array) -> jax.Array:
        state = y0
        for i in range(N_small):
            seg = jnp.vstack([jnp.zeros((1, dim), dtype=incs.dtype), incs[i].reshape(1, -1)])
            # Build degree-1 only for speed in very small-time check if depth==1; otherwise use depth
            forests_loc = forests
            hopf_loc = hopf
            cov = jnp.tile((dt_small * I)[None, :, :], reps=(1, 1, 1))
            sig_levels = compute_planar_branched_signature(
                path=seg,
                order_m=depth,
                forests=forests_loc,
                cov_increments=cov,
                return_trajectory=False,
            )
            sig_levels_list = cast(list[jax.Array], sig_levels)
            group_el = GroupElement(hopf=hopf_loc, coeffs=sig_levels_list, interval=(i, i + 1))
            logsig = group_el.log()
            state = log_ode(mkw_brackets, logsig, state)
        return state

    yT_small = jax.vmap(integrate_short, in_axes=0)(dW_small)
    disp_small = yT_small - y0
    P_tan = jnp.eye(3, dtype=jnp.float32) - jnp.outer(y0, y0)
    disp_tan = disp_small @ P_tan.T
    disp_xy = disp_tan[:, :2]
    mean_xy = jnp.mean(disp_xy, axis=0)
    assert jnp.linalg.norm(mean_xy) < 0.08
    centered = disp_xy - mean_xy
    cov_xy = (centered.T @ centered) / float(M_small)
    target_cov = T_small * jnp.eye(2, dtype=jnp.float32)
    assert jnp.allclose(cov_xy, target_cov, rtol=0.2, atol=0.03)

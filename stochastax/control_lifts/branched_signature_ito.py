from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from stochastax.hopf_algebras.hopf_algebra_types import (
    HopfAlgebra,
    BCKForest,
    MKWForest,
    GLHopfAlgebra,
    MKWHopfAlgebra,
)


def _zero_coeffs_from_hopf(hopf: HopfAlgebra, depth: int, dtype: jnp.dtype) -> list[jax.Array]:
    return [jnp.zeros((hopf.basis_size(i),), dtype=dtype) for i in range(depth)]


def local_ito_character(
    delta_x: jax.Array,
    cov: Optional[jax.Array],
    order_m: int,
    hopf: GLHopfAlgebra | MKWHopfAlgebra,
    extra: Optional[dict[int, float]] = None,
) -> list[jax.Array]:
    """Build per-step infinitesimal character for Itô branched signature."""
    if order_m != hopf.max_degree:
        raise ValueError("order_m must equal hopf.max_degree")
    d = hopf.ambient_dimension
    dtype = delta_x.dtype
    out = _zero_coeffs_from_hopf(hopf, order_m, dtype)

    # Degree 1: single-node tree with colour i gets delta_x[i]
    # We assume shape-major, then colour-lexicographic layout; colours for shape 0 occupy indices 0..d-1.
    out[0] = out[0].at[jnp.arange(d)].set(delta_x)

    # Degree 2: Itô correction on chain-of-length-2
    if order_m >= 2 and cov is not None:
        if hopf.degree2_chain_indices is None:
            raise ValueError("Degree-2 chain indices not available in Hopf algebra.")
        idx = hopf.degree2_chain_indices  # shape (d, d) of flattened indices
        updates = jnp.zeros_like(out[1])
        updates = updates.at[idx].set(cov)
        out[1] = updates

    # Degree >= 3: optional overrides (sparse)
    if extra:
        # The interpretation of keys is left to the caller; no offsets are stored here.
        # For safety, we ignore extras unless future extensions provide per-degree maps.
        pass

    return out


def _concat_levels(levels: list[jax.Array]) -> jax.Array:
    return jnp.concatenate(levels, axis=0) if levels else jnp.zeros((0,), dtype=jnp.float32)


def _branched_signature_ito_impl(
    path: jax.Array,
    order_m: int,
    hopf: GLHopfAlgebra | MKWHopfAlgebra,
    cov_increments: Optional[jax.Array] = None,
    higher_local_moments: Optional[list[dict[int, float]]] = None,
    return_trajectory: bool = False,
) -> list[jax.Array] | tuple[list[jax.Array], jax.Array]:
    """Compute the Itô branched signature along a sampled path (shared implementation)."""
    if path.ndim != 2:
        raise ValueError(f"Expected path of shape [T, d], got {path.shape}")
    T, d = path.shape
    if order_m <= 0:
        raise ValueError("order_m must be >= 1")
    if T <= 1:
        dtype = path.dtype
        S0 = _zero_coeffs_from_hopf(hopf, order_m, dtype)
        if return_trajectory:
            total_dim = sum(hopf.basis_size(i) for i in range(order_m))
            return S0, jnp.zeros((T, total_dim), dtype=dtype)
        return S0

    if hopf.max_degree != order_m:
        raise ValueError("forests must cover degrees 1..order_m (exact).")

    dtype = path.dtype
    sig = _zero_coeffs_from_hopf(hopf, order_m, dtype)  # unit (tail)

    traj: Optional[list[jax.Array]] = [] if return_trajectory else None
    total_dim = sum(hopf.basis_size(i) for i in range(order_m))

    for k in range(T - 1):
        delta_x = path[k + 1] - path[k]
        cov = cov_increments[k] if cov_increments is not None else None
        extra = higher_local_moments[k] if higher_local_moments is not None else None
        a_k = local_ito_character(delta_x, cov, order_m, hopf, extra)
        E_k = hopf.exp(a_k)
        sig = hopf.product(sig, E_k)
        if traj is not None:
            traj.append(_concat_levels(sig))

    if traj is not None:
        return sig, jnp.stack(traj, axis=0)
    return sig


def compute_planar_branched_signature(
    path: jax.Array,
    order_m: int,
    forests: list[MKWForest],
    cov_increments: Optional[jax.Array] = None,
    higher_local_moments: Optional[list[dict[int, float]]] = None,
    return_trajectory: bool = False,
) -> list[jax.Array] | tuple[list[jax.Array], jax.Array]:
    """Planar (MKW) Itô branched signature wrapper."""
    if path.ndim != 2:
        raise ValueError(f"Expected path of shape [T, d], got {path.shape}")
    d = int(path.shape[1])
    hopf = MKWHopfAlgebra.build(d, forests)
    return _branched_signature_ito_impl(
        path=path,
        order_m=order_m,
        hopf=hopf,
        cov_increments=cov_increments,
        higher_local_moments=higher_local_moments,
        return_trajectory=return_trajectory,
    )


def compute_nonplanar_branched_signature(
    path: jax.Array,
    order_m: int,
    forests: list[BCKForest],
    cov_increments: Optional[jax.Array] = None,
    higher_local_moments: Optional[list[dict[int, float]]] = None,
    return_trajectory: bool = False,
) -> list[jax.Array] | tuple[list[jax.Array], jax.Array]:
    """Nonplanar (BCK/GL) Itô branched signature wrapper."""
    if path.ndim != 2:
        raise ValueError(f"Expected path of shape [T, d], got {path.shape}")
    d = int(path.shape[1])
    hopf = GLHopfAlgebra.build(d, forests)
    return _branched_signature_ito_impl(
        path=path,
        order_m=order_m,
        hopf=hopf,
        cov_increments=cov_increments,
        higher_local_moments=higher_local_moments,
        return_trajectory=return_trajectory,
    )


if __name__ == "__main__":
    # Minimal example: compare standard tensor signature with branched Itô signature (m=2).
    import jax.numpy as jnp
    from typing import cast
    from stochastax.hopf_algebras import enumerate_bck_trees, enumerate_mkw_trees
    from stochastax.control_lifts.path_signature import compute_path_signature
    from stochastax.hopf_algebras.hopf_algebra_types import GLHopfAlgebra

    # Build a simple 2D path
    path = jnp.array(
        [
            [0.0, 0.0],
            [1.0, -0.5],
            [1.7, 0.2],
            [2.0, 0.8],
        ],
        dtype=jnp.float32,
    )
    depth = 2
    d = int(path.shape[1])

    # Standard (shuffle/tensor) signature
    std = compute_path_signature(path, depth=depth, mode="full")
    std_levels = std.coeffs  # list[jax.Array], levels 1..depth

    # Branched Itô signature on BCK and MKW trees up to degree 2
    bck_forests_list = enumerate_bck_trees(depth)  # list of BCKForest for degrees 1..depth
    mkw_forests_list = enumerate_mkw_trees(depth)  # list of MKWForest for degrees 1..depth
    increments = path[1:, :] - path[:-1, :]
    cov_zero = jnp.zeros((increments.shape[0], d, d), dtype=path.dtype)
    cov_dx_dx = jnp.einsum("td,te->tde", increments, increments)

    ito_zero_levels = cast(
        list[jax.Array],
        compute_nonplanar_branched_signature(
            path=path,
            order_m=depth,
            forests=bck_forests_list,
            cov_increments=cov_zero,
            return_trajectory=False,
        ),
    )
    ito_dx_dx_levels = cast(
        list[jax.Array],
        compute_planar_branched_signature(
            path=path,
            order_m=depth,
            forests=mkw_forests_list,
            cov_increments=cov_dx_dx,
            return_trajectory=False,
        ),
    )

    # Compare level-1 (should match total increment)
    lvl1_diff = jnp.linalg.norm(std_levels[0] - ito_zero_levels[0])
    print(f"Level-1 difference norm (standard vs branched Itô, cov=0): {float(lvl1_diff):.6f}")

    # Inspect degree-2
    print(f"Standard signature level-2 norm: {float(jnp.linalg.norm(std_levels[1])):.6f}")
    print(f"Branched Itô (cov=0) level-2 norm: {float(jnp.linalg.norm(ito_zero_levels[1])):.6f}")
    print(
        f"Branched Itô (cov=Δx⊗Δx) level-2 norm: {float(jnp.linalg.norm(ito_dx_dx_levels[1])):.6f}"
    )

    # Show chain-of-length-2 matrix entries for the BCK hopf
    bck_hopf = GLHopfAlgebra.build(d, bck_forests_list)
    if bck_hopf.degree2_chain_indices is not None:
        chain_zero = ito_zero_levels[1][bck_hopf.degree2_chain_indices]
        chain_dx_dx = ito_dx_dx_levels[1][bck_hopf.degree2_chain_indices]
        print("Branched Itô chain (cov=0):")
        print(chain_zero)
        print("Branched Itô chain (cov=Δx⊗Δx):")
        print(chain_dx_dx)

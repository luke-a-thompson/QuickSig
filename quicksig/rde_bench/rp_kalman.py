import jax
import jax.numpy as jnp
import diffrax as dfx
from quicksig.rde_bench.rough_volatility import make_rough_bergomi_model_spec, BonesiniModelSpec, build_terms_with_leadlag
from dataclasses import dataclass, replace

key = jax.random.key(42)

@jax.tree_util.register_pytree_node_class
@dataclass(slots=True)
class RBergomiEnsemble:
    # State (batched across N particles)
    S: jax.Array  # shape (N,)
    V: jax.Array  # shape (N,)

    # Unconstrained parameters per particle (batched)
    tH: jax.Array  # logit(H),   shape (N,)
    tEta: jax.Array  # log(eta),   shape (N,)
    tRho: jax.Array  # atanh(rho), shape (N,)
    tV0: jax.Array  # log(v0),    shape (N,)

    def tree_flatten(self):
        leaves = (self.S, self.V, self.tH, self.tEta, self.tRho, self.tV0)
        aux = None
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        S, V, tH, tEta, tRho, tV0 = leaves
        return cls(S, V, tH, tEta, tRho, tV0)

    def replace(self, **kwargs):
        return replace(self, **kwargs)

particle_count = 100
key_nu, key_rho, key_hurst = jax.random.split(key, 3)
nu_dist = jnp.log(1.5) + 0.5**2 * jax.random.normal(key_nu, (particle_count,))
rho_dist = -0.7 + 0.5**2 * jax.random.normal(key_rho, (particle_count,))
hurst_dist = 0.165 * jax.random.gamma(key_hurst, 2.0, (particle_count,))

# Create multiple rBergomi model specifications
def create_rbergomi_ensemble_specs(nu_params: jax.Array, rho_params: jax.Array, hurst_params: jax.Array, v0: float) -> list[BonesiniModelSpec]:
    """Create multiple rBergomi model specifications from parameter arrays."""
    specs = []
    for i in range(len(nu_params)):
        spec = make_rough_bergomi_model_spec(
            v_0=v0,
            nu=float(nu_params[i]),
            hurst=float(hurst_params[i]),
            rho=float(rho_params[i]),
        )
        specs.append(spec)
    return specs


def create_terms_with_leadlag(model_specs: list[BonesiniModelSpec], Z_control: dfx.LinearInterpolation) -> list[dfx.MultiTerm]:
    return [build_terms_with_leadlag(spec, Z_control=Z_control) for spec in model_specs]

solver = dfx.Heun()







def step_rough_volatility_ensemble(state: RBergomiEnsemble, vector_fields: list[dfx.MultiTerm], dt: float) -> RBergomiEnsemble:

# Create the ensemble of model specifications
model_specs = create_rbergomi_ensemble_specs(nu_dist, rho_dist, hurst_dist, v0=0.04)

state = solver.init()






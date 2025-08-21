import jax
import jax.numpy as jnp
import diffrax as dfx
from quicksig.rde_bench.rough_volatility import make_rough_bergomi_model_spec, BonesiniModelSpec
from dataclasses import dataclass, replace

key = jax.random.key(42)

## Set up particle ensemble
# params = (nu, rho, hurst)
ensemble_size = 1000
key_nu, key_rho, key_hurst = jax.random.split(key, 3)
nu_dist = jnp.log(1.5) + 0.5**2 * jax.random.normal(key_nu, (ensemble_size,))
rho_dist = -0.7 + 0.5**2 * jax.random.normal(key_rho, (ensemble_size,))
hurst_dist = jax.random.gamma(key_hurst, 2.0, (ensemble_size,)) * 0.165

solver = dfx.Heun()
vector_fields = 
def rough_volatility_state_space_model(model_spec: BonesiniModelSpec, W_control: dfx.LinearInterpolation, X_control: dfx.LinearInterpolation):
    dfx.Heun()

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

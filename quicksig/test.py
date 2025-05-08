import jax
import jax.numpy as jnp
from functools import partial
import lovely_jax as lj

lj.monkey_patch()




if __name__ == "__main__":
    from quicksig.path_signature import batch_signature_pure_jax

    key = jax.random.PRNGKey(0)
    path = jax.random.normal(key, shape=(1, 100, 4))
    signature = batch_signature_pure_jax(path, depth=5)
    print(signature.shape)
    log_signature = flat_tensor_log(signature, 5, 4)
    print(log_signature.shape)

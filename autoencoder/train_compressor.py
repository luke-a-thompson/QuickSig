import sys
import os

# Add current directory to path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add parent directory to path for proper imports
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

try:
    # Try relative import first (when run as module)
    from .autoencoder import LogsigCompressor, LogsigDecompressor
except ImportError:
    # Fall back to direct import from the same directory (when run directly)
    import autoencoder

    LogsigCompressor = autoencoder.LogsigCompressor
    LogsigDecompressor = autoencoder.LogsigDecompressor

from quicksig import get_log_signature


# Helper function to calculate sum of squared L2 norms for model parameters
def _get_param_norm_sq(pytree: eqx.Module) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(eqx.filter(pytree, eqx.is_inexact_array))
    # Ensure a JAX array is returned, even if leaves is empty or sum is 0.
    return jnp.array(sum(jnp.sum(leaf**2) for leaf in leaves if leaf is not None), dtype=jnp.float32)


@eqx.filter_jit
def _compute_loss(
    models: tuple[LogsigCompressor, LogsigDecompressor],
    low_depth_logsig: jax.Array,
    high_depth_logsig: jax.Array,
    C_AE: float,
    C_e: float,
) -> jax.Array:  # Returns only the scalar loss
    compressor, decompressor = models

    compressed_logsig = compressor(low_depth_logsig[1])

    reconstructed_logsig = decompressor(compressed_logsig)

    L_recon = jnp.linalg.norm(reconstructed_logsig - high_depth_logsig, ord=2)

    comp_param_norm_sq = _get_param_norm_sq(compressor)
    decomp_param_norm_sq = _get_param_norm_sq(decompressor)

    reconstructed_high_depth_logsig_norm_sq = jnp.sum(reconstructed_logsig**2)

    regularization_loss = C_AE * (comp_param_norm_sq + decomp_param_norm_sq) + C_e * reconstructed_high_depth_logsig_norm_sq

    L_AE = L_recon + regularization_loss
    return L_AE


@eqx.filter_jit
def train_step(
    models: tuple[LogsigCompressor, LogsigDecompressor],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    low_depth_logsig: jax.Array,
    high_depth_logsig: jax.Array,
    C_AE: float,
    C_e: float,
) -> tuple[jax.Array, tuple[LogsigCompressor, LogsigDecompressor], optax.OptState]:

    # Define a closure for the loss function that only takes 'models' as input
    def loss_fn_for_grad(current_models: tuple[LogsigCompressor, LogsigDecompressor]) -> jax.Array:
        return _compute_loss(current_models, low_depth_logsig, high_depth_logsig, C_AE, C_e)

    loss_value, grads = eqx.filter_value_and_grad(loss_fn_for_grad)(models)

    filtered_models_for_update = eqx.filter(models, eqx.is_inexact_array)
    updates, new_opt_state = optimizer.update(grads, opt_state, filtered_models_for_update)

    new_models = eqx.apply_updates(models, updates)

    return loss_value, new_models, new_opt_state


def main(epochs: int, C_AE: float = 1.0, C_e: float = 1) -> None:
    key = jax.random.PRNGKey(seed=0)
    path_key, compressor_key, decompressor_key = jax.random.split(key, 3)

    # Create a batch of 1 path: (batch_size, sequence_length, num_channels)
    batched_path = jax.random.normal(path_key, shape=(1, 200, 5))

    # get_log_signature will return (batch_size, num_features)
    low_depth_logsig: jax.Array = get_log_signature(batched_path, depth=2, log_signature_type="lyndon")
    high_depth_logsig: jax.Array = get_log_signature(batched_path, depth=3, log_signature_type="lyndon")

    compressor = LogsigCompressor(
        low_depth_logsig_dim=low_depth_logsig.shape[1],
        hidden_dim=low_depth_logsig.shape[1] * 2,
        compressed_dim=low_depth_logsig.shape[1],
        key=compressor_key,
    )
    decompressor = LogsigDecompressor(
        compressed_dim=low_depth_logsig.shape[1],
        hidden_dim=high_depth_logsig.shape[1] * 2,
        high_depth_logsig_dim=high_depth_logsig.shape[1],
        key=decompressor_key,
    )

    models = (compressor, decompressor)

    optimizer = optax.adam(learning_rate=0.001)
    # Initialize optimizer state with the filtered Pytree of trainable parameters
    opt_state = optimizer.init(eqx.filter(models, eqx.is_inexact_array))

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        loss, models, opt_state = train_step(models, opt_state, optimizer, low_depth_logsig, high_depth_logsig, C_AE, C_e)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print("Training finished.")


if __name__ == "__main__":
    main(epochs=20000)

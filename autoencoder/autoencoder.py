import jax
import equinox as eqx


class LogsigCompressor(eqx.Module):
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    rmsnorm: eqx.nn.RMSNorm

    def __init__(
        self,
        low_depth_logsig_dim: int,
        hidden_dim: int,
        compressed_dim: int,
        key: jax.Array,
    ) -> None:
        super().__init__()
        mlp_key, output_key = jax.random.split(key)

        self.linear_in = eqx.nn.Linear(
            in_features=low_depth_logsig_dim,
            out_features=hidden_dim,
            key=mlp_key,
        )
        self.linear_out = eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=compressed_dim,
            key=output_key,
        )

        self.rmsnorm = eqx.nn.RMSNorm(
            shape=(hidden_dim,),
        )

    def __call__(self, logsig: jax.Array) -> jax.Array:
        x = self.linear_in(logsig)
        x = jax.nn.gelu(x)
        x = self.rmsnorm(x)
        x = self.linear_out(x)

        return x


class LogsigDecompressor(eqx.Module):
    linear_in: eqx.nn.Linear
    linear_hidden: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    rmsnorm1: eqx.nn.RMSNorm
    rmsnorm2: eqx.nn.RMSNorm

    def __init__(
        self,
        compressed_dim: int,
        hidden_dim: int,
        high_depth_logsig_dim: int,
        key: jax.Array,
    ) -> None:
        super().__init__()
        key_in, key_hidden, key_out = jax.random.split(key, 3)

        self.linear_in = eqx.nn.Linear(
            in_features=compressed_dim,
            out_features=hidden_dim,
            key=key_in,
        )
        self.linear_hidden = eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            key=key_hidden,
        )
        self.linear_out = eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=high_depth_logsig_dim,
            key=key_out,
        )

        self.rmsnorm1 = eqx.nn.RMSNorm(
            shape=(hidden_dim,),
        )
        self.rmsnorm2 = eqx.nn.RMSNorm(
            shape=(hidden_dim,),
        )

    def __call__(self, compressed_logsig: jax.Array) -> jax.Array:
        x = self.linear_in(compressed_logsig)
        x = jax.nn.gelu(x)
        x = self.rmsnorm1(x)
        x = self.linear_hidden(x)
        x = jax.nn.gelu(x)
        x = self.rmsnorm2(x)
        x = self.linear_out(x)

        return x

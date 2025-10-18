import jax
import jax.numpy as jnp
import pytest

from quicksig.rde.rde_types import Path, pathify


@pytest.mark.parametrize("stream_shape", [(100, 3), (3, 100, 3)])
def test_path_concatenation_success(stream_shape: tuple[int, ...]):
    stream = jax.random.normal(jax.random.PRNGKey(42), stream_shape)
    path = pathify(stream)
    split_paths = path.split_at_time([33, 66])

    assert split_paths[0] + split_paths[1] + split_paths[2] == path


@pytest.mark.parametrize("stream_shape", [(100, 3), (3, 100, 3)])
def test_path_concatenation_failure(stream_shape: tuple[int, ...]):
    stream = jax.random.normal(jax.random.PRNGKey(42), stream_shape)
    path = pathify(stream)
    split_paths = path.split_at_time([33, 66])

    with pytest.raises(ValueError):
        _ = split_paths[0] + split_paths[2]
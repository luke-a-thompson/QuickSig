import jax
import jax.numpy as jnp
import pytest

from quicksig.controls.paths_types import Path, pathify


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


def test_path_properties_error_cases():
    """Test Path properties with invalid dimensions."""
    # Invalid ndim for num_timesteps
    invalid_1d = jnp.array([1, 2, 3])
    path_1d = Path(path=invalid_1d, interval=(0, 3))
    with pytest.raises(ValueError, match="Path must be 2D or 3D"):
        _ = path_1d.num_timesteps

    # Invalid ndim for ambient_dimension
    with pytest.raises(ValueError, match="Path must be 2D or 3D"):
        _ = path_1d.ambient_dimension


def test_path_getitem_with_stride():
    """Test that slicing with stride raises ValueError."""
    stream = jax.random.normal(jax.random.PRNGKey(42), (100, 3))
    path = pathify(stream)

    with pytest.raises(ValueError, match="Slicing with a step is not supported"):
        _ = path[::2]


def test_path_getitem_negative_index():
    """Test negative indexing for paths."""
    stream = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    path = pathify(stream)

    # Negative index should work
    last_path = path[-1]
    assert last_path.interval == (3, 3)


def test_path_str_and_repr():
    """Test __str__ and __repr__ methods."""
    stream = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    path = pathify(stream)

    str_repr = str(path)
    assert "Path(" in str_repr
    assert "interval=(0, 3)" in str_repr
    assert "num_timesteps=3" in str_repr
    assert "ambient_dimension=2" in str_repr

    # __repr__ should be the same as __str__
    assert repr(path) == str(path)


def test_path_eq_type_error():
    """Test __eq__ with non-Path objects."""
    stream = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    path = pathify(stream)

    with pytest.raises(NotImplementedError, match="Cannot compare Path with"):
        _ = path == "not a path"

    with pytest.raises(NotImplementedError, match="Cannot compare Path with"):
        _ = path == 42


def test_path_add_dimension_mismatch():
    """Test __add__ with mismatched ambient dimensions."""
    path1 = Path(path=jnp.array([[1.0, 2.0], [3.0, 4.0]]), interval=(0, 2))
    path2 = Path(path=jnp.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), interval=(2, 4))

    with pytest.raises(ValueError, match="Paths must have the same ambient dimension"):
        _ = path1 + path2


def test_path_add_ndim_mismatch():
    """Test __add__ with mismatched ndim."""
    path1 = Path(path=jnp.array([[1.0, 2.0], [3.0, 4.0]]), interval=(0, 2))
    path2_3d = Path(path=jnp.array([[[5.0, 6.0], [7.0, 8.0]]]), interval=(2, 4))

    with pytest.raises(ValueError, match="Paths must have the same number of dimensions"):
        _ = path1 + path2_3d


def test_pathify_invalid_ndim():
    """Test pathify with invalid array dimensions."""
    # 1D array should raise ValueError
    invalid_1d = jnp.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="Stream must be a 2D or 3D array"):
        pathify(invalid_1d)

    # 4D array should raise ValueError
    invalid_4d = jnp.ones((2, 3, 4, 5))
    with pytest.raises(ValueError, match="Stream must be a 2D or 3D array"):
        pathify(invalid_4d)

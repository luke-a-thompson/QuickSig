import pytest
from quicksig.analytics.signature_sizes import get_signature_dim


@pytest.mark.parametrize(
    "depth, dim, expected_dim",
    [
        (1, 1, 1),
        (3, 2, 14),
        (2, 5, 30),
        (4, 1, 4),
        (1, 10, 10),
    ],
)
def test_get_signature_dim_flatten(depth: int, dim: int, expected_dim: int) -> None:
    """Test get_signature_dim with flatten=True."""
    assert get_signature_dim(depth, dim, flatten=True) == expected_dim


@pytest.mark.parametrize(
    "depth, dim, expected_dims",
    [
        (1, 1, [1]),
        (3, 2, [2, 4, 8]),
        (2, 5, [5, 25]),
        (4, 1, [1, 1, 1, 1]),
        (1, 10, [10]),
    ],
)
def test_get_signature_dim_not_flatten(depth: int, dim: int, expected_dims: list[int]) -> None:
    """Test get_signature_dim with flatten=False."""
    assert get_signature_dim(depth, dim, flatten=False) == expected_dims

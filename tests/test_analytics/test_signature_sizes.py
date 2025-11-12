import pytest
from quicksig.analytics.signature_sizes import (
    get_signature_dim,
    get_log_signature_dim,
    get_bck_signature_dim,
    get_mkw_signature_dim,
)


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
    assert get_signature_dim(depth, dim) == expected_dim


# --------------------------
# Log-signature (Witt/Lyndon) – expected counts
# --------------------------


@pytest.mark.parametrize(
    "depth, dim, expected_sum",
    [
        (6, 2, 23),  # 2+1+2+3+6+9
        (5, 3, 80),  # 3+3+8+18+48
        (4, 1, 1),  # 1+0+0+0
    ],
)
def test_get_log_signature_dim_flatten(depth: int, dim: int, expected_sum: int) -> None:
    assert get_log_signature_dim(depth, dim) == expected_sum


# --------------------------
# BCK (unordered rooted forests) – OEIS A000081
# --------------------------


@pytest.mark.parametrize(
    "depth, dim, expected_sum",
    [
        (6, 1, 84),  # sum([1,2,4,9,20,48])
        (5, 2, 826),  # sum([2,8,32,144,640])
    ],
)
def test_get_bck_signature_dim_flatten(depth: int, dim: int, expected_sum: int) -> None:
    assert get_bck_signature_dim(depth, dim) == expected_sum


# --------------------------
# MKW (ordered/plane rooted forests) – Catalan numbers
# --------------------------


@pytest.mark.parametrize(
    "depth, dim, expected_sum",
    [
        (5, 1, 64),  # 1+2+5+14+42
        (4, 3, 1290),  # 3+18+135+1134
    ],
)
def test_get_mkw_signature_dim_flatten(depth: int, dim: int, expected_sum: int) -> None:
    assert get_mkw_signature_dim(depth, dim) == expected_sum

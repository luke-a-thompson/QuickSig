import pytest
from quicksig.signatures.compute_path_signature import compute_path_signature
from jax import random


paths = [
    pytest.param((100, 5), id="timesteps=100,features=5"),
    pytest.param((1000, 5), id="timesteps=1000,features=5"),
    pytest.param((10_000, 5), id="timesteps=10_000,features=5"),
    pytest.param((250_000, 5), id="timesteps=250_000,features=5"),
    pytest.param((500_000, 5), id="timesteps=500_000,features=5"),
    pytest.param((750_000, 5), id="timesteps=750_000,features=5"),
    pytest.param((1_000_000, 5), id="timesteps=1_000_000,features=5"),
    pytest.param((100, 10), id="timesteps=100,features=10"),
    pytest.param((1000, 10), id="timesteps=1000,features=10"),
    pytest.param((10_000, 10), id="timesteps=10_000,features=10"),
    pytest.param((250_000, 10), id="timesteps=250_000,features=10"),
    pytest.param((500_000, 10), id="timesteps=500_000,features=10"),
    pytest.param((750_000, 10), id="timesteps=750_000,features=10"),
    pytest.param((1_000_000, 10), id="timesteps=1_000_000,features=10"),
    pytest.param((100, 20), id="timesteps=100,features=20"),
    pytest.param((1000, 20), id="timesteps=1000,features=20"),
    pytest.param((10_000, 20), id="timesteps=10_000,features=20"),
    pytest.param((250_000, 20), id="timesteps=250_000,features=20"),
    pytest.param((500_000, 20), id="timesteps=500_000,features=20"),
    pytest.param((750_000, 20), id="timesteps=750_000,features=20"),
    pytest.param((1_000_000, 20), id="timesteps=1_000_000,features=20"),
]


# @pytest.mark.timeout(10)
# @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
# @pytest.mark.parametrize("scalar_path_fixture", paths)
# def test_memory_reqs(depth: int, scalar_path_fixture: tuple[int, int]):
#     """Test that the memory requirements are reasonable for the given depth and size."""
#     path = random.normal(random.key(0), scalar_path_fixture)
#     compute_path_signature(path, depth=depth, mode="full")

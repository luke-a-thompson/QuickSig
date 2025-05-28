import jax
import jax.numpy as jnp
from quicksig.log_signature import duval_algorithm


def test_duval_algorithm():
    # Test with depth=2 and dim=2 (binary case)
    result = duval_algorithm(depth=2, dim=2)

    # Print each level's words
    for i, words in enumerate(result):
        print(f"\nLevel {i+1} words:")
        print(words)

    # Expected output:
    # Level 1 words: [[0], [1]]  # Single letter words
    # Level 2 words: [[0, 1]]    # Two letter Lyndon words

    # Test with depth=3 and dim=2
    result = duval_algorithm(depth=3, dim=2)

    print("\nWith depth=3, dim=2:")
    for i, words in enumerate(result):
        print(f"\nLevel {i+1} words:")
        print(words)

    # Expected output:
    # Level 1 words: [[0], [1]]
    # Level 2 words: [[0, 1]]
    # Level 3 words: [[0, 0, 1], [0, 1, 1]]


if __name__ == "__main__":
    test_duval_algorithm()

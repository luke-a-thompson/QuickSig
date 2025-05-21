# QuickSig

An extremely simple, fast signature computation library in Jax.

## Installation

```bash
pip install quicksig
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
import jax.numpy as jnp
from quicksig import compute_path_signature, batch_compute_signatures

# Compute signature for a single path
path = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
signature = compute_path_signature(path, depth=2)

# Compute signatures for a batch of paths
paths = jnp.array([
    [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
    [[0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
])
signatures = batch_compute_signatures(paths, depth=2)
```

## Development

- Run tests: `pytest`
- Type checking: `mypy .`
- Format code: `black .`

## License

MIT

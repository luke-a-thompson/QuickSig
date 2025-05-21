import jax
from quicksig.batch_ops import batch_tensor_log
from quicksig.path_signature import batch_signature
from enum import StrEnum


class LogSignatureType(StrEnum):
    EXPANDED = "expanded"  # $$ \in T(V) $$
    LYNDON = "lyndon"


def batch_log_signature(path: jax.Array, depth: int, log_signature_type: LogSignatureType) -> list[jax.Array]:
    n_features = path.shape[-1]
    signature: list[jax.Array] = batch_signature(path, depth, stream=False)
    match log_signature_type:
        case LogSignatureType.EXPANDED:
            return batch_tensor_log(signature, n_features)
        case LogSignatureType.LYNDON:
            raise NotImplementedError("Lyndon log signature not implemented")
        case _:
            raise ValueError(f"Invalid log signature type: {log_signature_type}")

from typing import NewType
import jax

ButcherElementaryDifferential = NewType("ButcherElementaryDifferential", jax.Array)
LieButcherElementaryDifferential = NewType("LieButcherElementaryDifferential", jax.Array)

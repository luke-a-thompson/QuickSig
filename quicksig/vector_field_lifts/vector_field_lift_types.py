from typing import NewType
import jax

ButcherDifferentials = NewType("ButcherDifferentials", jax.Array)
LieButcherDifferentials = NewType("LieButcherDifferentials", jax.Array)

LyndonBrackets = NewType("LyndonBrackets", jax.Array)
ButcherBrackets = NewType("ButcherBrackets", jax.Array)
LieButcherBrackets = NewType("LieButcherBrackets", jax.Array)
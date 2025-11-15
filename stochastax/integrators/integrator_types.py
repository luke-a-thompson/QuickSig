from typing import NewType
import jax

ButcherSeries = NewType("ButcherSeries", jax.Array)
LieButcherSeries = NewType("LieButcherSeries", jax.Array)
LieSeries = NewType("LieSeries", jax.Array)

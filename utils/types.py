import enum
from typing import Any, Callable, Dict, NamedTuple, Sequence, Union

import numpy as np
from flax import core
import jax

Array = Union[np.ndarray, jax.Array]
DataDict = Dict[str, Array]

DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
InfoDict = Dict[str, float]
Params = Union[Dict, core.FrozenDict]
BatchStats = Any
ApplyFn = Callable[..., Any]
Shape = Sequence[int]

class ReductionType(enum.IntEnum):
    MIN = 0
    MEAN = 1

class Statistics(NamedTuple):
    mean: np.ndarray = None
    std: np.ndarray = None
    min: np.ndarray = None
    max: np.ndarray = None

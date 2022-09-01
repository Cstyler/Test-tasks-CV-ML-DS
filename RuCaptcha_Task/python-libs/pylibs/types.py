from pathlib import Path
from typing import Callable, Dict, Tuple, TypeVar, Union

from numpy import ndarray

try:
    from torch.nn import Module
except ImportError as e:
    Module = object

T = TypeVar('T')

Pathlike = Union[str, Path]
TupleStrType = Tuple[str, ...]
SplitDictType = Dict[str, Dict[str, Union[Tuple[float, float], float]]]
MetricFunctionType = Callable[[T, T], float]
Image = ndarray
TorchModel = Module
FilterFunction = Callable[[T], bool]

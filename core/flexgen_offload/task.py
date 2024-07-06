import numpy as np
import dataclasses
from typing import Union, List, Optional

@dataclasses.dataclass(frozen=True)
class Task:
    """A generation task."""
    inputs: Union[np.array, List[List[int]]]
    prompt_len: int
    gen_len: int
    cut_gen_len: Optional[int]

    do_sample: bool
    temperature: float
    stop: Optional[int]
    top_p: Optional[int] # top_p is not used in opt model
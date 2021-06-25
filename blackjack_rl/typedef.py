from typing import Tuple, List, TypeVar

State = Tuple[int, int, bool]
Trans = Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]
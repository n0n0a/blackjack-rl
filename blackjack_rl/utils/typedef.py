from typing import Tuple, Callable

State = Tuple[int, int, bool]
Action = bool
Reward = int
Trans = Tuple[State, Action, Reward, State]
Policy = Callable[[State], Action]
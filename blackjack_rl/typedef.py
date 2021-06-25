from typing import Tuple

State = Tuple[int, int, bool]
Action = bool
Reward = int
Trans = Tuple[State, Action, Reward, State]

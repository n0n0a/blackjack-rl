from blackjack_rl.agent import Agent
from typing import Tuple, List, Union
import numpy as np

state_space = (10, 9, 2)
action_space = (2)
all_space = state_space + action_space
all_zize = np.prod(all_space)


class LSPIAgent(Agent):
    def __init__(self):
        self.weight = np.zeros(all_zize)

    @staticmethod
    def translate_idx(self, state: Tuple[int, int, bool], action: bool) -> int:
        offset = 1
        idx = 0
        all = state + (action)
        for i, val in all:
            assert 0 <= val < all_space[i]
            idx += int(val) * offset
            offset *= all_space[i]
        return idx

    def take_action(self, state: Tuple[int, int, bool]) -> bool:
        if state[0] < 12:
            return True
        return self.weight[self.translate_idx(self, state=state, action=True)] \
               >= self.weight[self.translate_idx(self, state=state, action=False)]

    def train(self, data: Union[List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]],
                                Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]):
        raise NotImplementedError

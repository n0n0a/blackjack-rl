from abc import ABCMeta, abstractmethod
from typing import Tuple, List
from blackjack_rl.typedef import State, Trans


class Agent:
    @abstractmethod
    def take_action(self, state: State) -> bool:
        raise NotImplementedError

    @abstractmethod
    def train(self, train_data: List[Trans]):
        raise NotImplementedError

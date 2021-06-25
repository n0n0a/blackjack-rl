from abc import ABC, abstractmethod
from typing import Tuple, List
from blackjack_rl.typedef import State, Action, Trans


class Agent(ABC):
    @abstractmethod
    def take_action(cls, state: State) -> Action:
        raise NotImplementedError

    @abstractmethod
    def train(cls, train_data: List[Trans]):
        raise NotImplementedError

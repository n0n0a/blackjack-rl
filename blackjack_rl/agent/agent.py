from abc import ABC, abstractmethod
from typing import List
from blackjack_rl.utils.typedef import State, Action, Trans


class Agent(ABC):
    @abstractmethod
    def take_action(cls, state: State) -> Action:
        raise NotImplementedError

    @abstractmethod
    def train(cls, train_data: List[Trans]):
        raise NotImplementedError

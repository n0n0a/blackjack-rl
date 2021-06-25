from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union


class Agent:
    @abstractmethod
    def take_action(self, state: Tuple[int, int, bool]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def train(self, data: Union[List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]],
                               Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]):
        raise NotImplementedError

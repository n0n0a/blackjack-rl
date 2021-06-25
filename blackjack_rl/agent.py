from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union


class Agent(ABCMeta):
    @abstractmethod
    def take_action(cls) -> bool:
        raise NotImplementedError

    @abstractmethod
    def train(cls, data: Union[List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]],
                               Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]):
        raise NotImplementedError

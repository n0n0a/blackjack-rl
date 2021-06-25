from abc import ABCMeta, abstractmethod


class Agent(ABCMeta):
    @abstractmethod
    def take_action(cls):
        raise NotImplementedError

    @abstractmethod
    def train(cls, data):
        raise NotImplementedError

from abc import ABC, abstractmethod


class Policy(ABC):
    @property
    @abstractmethod
    def batch_size(self):
        pass

    @abstractmethod
    def __call__(self, states):
        pass

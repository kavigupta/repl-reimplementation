from abc import ABC, abstractmethod


class Driver(ABC):
    @abstractmethod
    def select(self, elements):
        pass


class DriverError(Exception):
    pass

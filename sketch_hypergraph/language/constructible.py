from abc import ABC, abstractmethod


class Constructible(ABC):
    def node_summary(self):
        """
        Summary of just this node
        """
        return f"{self.node_class()}({', '.join(repr(x) for x in self.node_params())})"

    @abstractmethod
    def node_class(self):
        """
        Class this this node belongs to. E.g., "Constant", "Variable", "NBinOp" etc.
        """
        pass

    @abstractmethod
    def node_params(self):
        """
        Parameters of this node. Non-leaf nodes do not have parameters; leaf nodes do.
        """
        pass

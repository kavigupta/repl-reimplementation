from abc import ABC, abstractmethod

import attr
import torch


class Distribution(ABC):
    @abstractmethod
    def mle(self):
        pass

    @abstractmethod
    def log_probability(self, outcomes):
        pass


@attr.s
class JointClassDistribution(Distribution):
    type = attr.ib()
    by_parameter = attr.ib()

    def mle(self):
        count = len(next(iter(self.by_parameter.values())))
        max_each = {
            param: self.by_parameter[param].max(1)[1].cpu().numpy()
            for param in self.by_parameter
        }
        return [
            self.type(**{k: v[i] for k, v in max_each.items()}) for i in range(count)
        ]

    def log_probability(self, outcomes):
        log_probs = []
        for param in self.by_parameter:
            values = [getattr(outcome, param) for outcome in outcomes]
            log_probs.append(self.by_parameter[param][range(len(values)), values].sum())
        return sum(log_probs)

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

    @abstractmethod
    def sample(self, rng):
        pass


@attr.s
class IndependentDistribution(Distribution):
    type = attr.ib()
    by_parameter = attr.ib()
    getattr = attr.ib(default=getattr)

    @property
    def _count(self):
        return len(next(iter(self.by_parameter.values())))

    def _initialize(self, params_each):
        return [
            self.type(**{k: v[i].item() for k, v in params_each.items()})
            for i in range(self._count)
        ]

    def mle(self):
        max_each = {
            param: self.by_parameter[param].max(1)[1].cpu().numpy()
            for param in self.by_parameter
        }
        return self._initialize(max_each)

    def sample(self, rng):
        sample_each = {}
        for param in self.by_parameter:
            old_seed = torch.random.get_rng_state()
            torch.random.manual_seed(rng.randint(2 ** 32))
            sample = torch.distributions.Categorical(
                logits=self.by_parameter[param]
            ).sample()
            torch.random.set_rng_state(old_seed)
            sample_each[param] = sample
        return self._initialize(sample_each)

    def log_probability(self, outcomes):
        log_probs = []
        for param in self.by_parameter:
            values = [self.getattr(outcome, param) for outcome in outcomes]
            log_probs.append(self.by_parameter[param][range(len(values)), values])
        return torch.stack(log_probs).sum(0)

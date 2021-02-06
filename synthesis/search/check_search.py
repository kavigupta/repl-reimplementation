import attr

from .search import Search
from ..environment.executor import evaluate


@attr.s
class CheckSearch(Search):
    executor = attr.ib()
    underlying = attr.ib()

    def __call__(self, m, spec):
        results = self.underlying(m, spec)
        data = [
            ((self.score(program, spec), weight), program)
            for weight, program in results
        ]
        return sorted(
            data,
            reverse=True,
        )

    def score(self, program, spec):
        return evaluate(self.executor, program, spec, use_test=False)["correct"]

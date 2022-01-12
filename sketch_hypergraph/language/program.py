import attr

from .sampler import SamplerError
from .statement import Statement


def sample_program(sampler, typenv):
    while True:
        try:
            return Program.sample(sampler, typenv)
        except SamplerError as e:
            print(f"Error in sampling: {e}; retrying...")


@attr.s
class Program:
    statements = attr.ib()
    starting_typeenv = attr.ib()

    @classmethod
    def sample(cls, sampler, typenv):
        length = sampler.sample_length()
        statements = []
        for _ in range(length):
            statement, typenv = Statement.sample(sampler, typenv)

            statements.append(statement)
        return cls(statements, typenv), typenv

import attr

# TODO handle tree structured programs
@attr.s
class Program:
    tokens = attr.ib()

    @property
    def partials(self):
        for t in range(len(self.tokens)):
            yield [Program(self.tokens[:t])], self.tokens[t]

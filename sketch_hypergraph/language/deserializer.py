from sketch_hypergraph.language.driver import Driver


class DeSerializingDriver(Driver):
    def __init__(self, sequence):
        self.sequence = list(reversed(sequence))

    def select(self, elements):
        tags = [el.node_summary() for el in elements]
        assert len(set(tags)) == len(tags)
        token = self.sequence.pop()
        return elements[tags.index(token)]

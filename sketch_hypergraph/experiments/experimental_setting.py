import attr


@attr.s
class ExperimentalSetting:
    context = attr.ib()
    grammar = attr.ib()
    value_grammar = attr.ib()
    sampler_spec = attr.ib()

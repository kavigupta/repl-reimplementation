import attr

from .ast import ASTNode, BlockNode, ForNode, IfNode
from .value import Number
from .sketch_hypergraph_grammar import (
    BOOLEAN_CONDITION_OPERATION,
    BOOLEAN_ARITHMETIC_OPERATION,
    NUMERIC_ARITHMETIC_OPERATION,
)


@attr.s
class EvaluationResult:
    value = attr.ib()
    drawn_objects = attr.ib()
    new_environment = attr.ib()


@attr.s
class EvaluationContext:
    statement_signatures = attr.ib()

    @classmethod
    def of(cls, *elements):
        return cls({element.name: element for element in elements})

    def post_node_environment(self, node, pre_node_environment):
        if isinstance(node, ForNode):
            return pre_node_environment

        if isinstance(node, IfNode):
            then_post = self.post_node_environment(
                node.then_branch, pre_node_environment
            )
            else_post = self.post_node_environment(
                node.else_branch, pre_node_environment
            )
            return then_post.intersection(else_post)

        if isinstance(node, BlockNode):
            environment = pre_node_environment
            for element in node.elements:
                environment = self.post_node_environment(element, environment)
            return environment
        assert isinstance(node, ASTNode)
        signature = self.statement_signatures[node.tag].bind(node)

        return signature.bind_outputs(pre_node_environment)


@attr.s
class Environment:
    evaluation_context = attr.ib()
    variables = attr.ib()
    drawn_objects = attr.ib(default=attr.Factory(list))

    def evaluate(self, x):
        return dict(
            Constant=lambda env, c: EvaluationResult(Number(c.value), [], env),
            Variable=lambda env, v: EvaluationResult(self.variables[v.name], [], env),
            NBinop=type(self).evaluate_op,
            BNBinop=type(self).evaluate_op,
            BBBinop=type(self).evaluate_op,
            BUnop=type(self).evaluate_op,
            Range=type(self).evaluate_range,
            Block=type(self).evaluate_block,
            If=type(self).evaluate_if,
            For=type(self).evaluate_for,
            **{
                name: c.evaluate
                for name, c in self.evaluation_context.statement_signatures.items()
            },
        )[x.node_class()](self, x)

    def bind(self, var, val):
        return Environment(
            self.evaluation_context, {**self.variables, var: val}, self.drawn_objects
        )

    def evaluate_op(self, x):
        symbol, *operands = x.children
        operand_results = [self.evaluate(operand) for operand in operands]
        operand_values = [res.value.value for res in operand_results]
        all_objects = [obj for res in operand_results for obj in res.drawn_objects]
        assert all(res.new_environment is self for res in operand_results)
        operation = {
            **NUMERIC_ARITHMETIC_OPERATION,
            **BOOLEAN_ARITHMETIC_OPERATION,
            **BOOLEAN_CONDITION_OPERATION,
            "!": lambda x: not x,
        }[symbol.symbol]
        return EvaluationResult(Number(operation(*operand_values)), all_objects, self)

    def evaluate_range(self, x):
        start, end, step = x.children
        start_res = self.evaluate(start)
        end_res = self.evaluate(end)
        step_res = self.evaluate(step)
        assert (
            self
            is start_res.new_environment
            is end_res.new_environment
            is step_res.new_environment
        )
        start, end, step = (
            start_res.value.value,
            end_res.value.value,
            step_res.value.value,
        )
        return EvaluationResult(
            range(start, end, step) if step != 0 else [],
            start_res.drawn_objects + end_res.drawn_objects + step_res.drawn_objects,
            self,
        )

    def evaluate_block(self, x):
        env = self
        overall = []
        for element in x.elements:
            res = env.evaluate(element)
            overall += res.drawn_objects
            env = res.new_environment
        return EvaluationResult(None, overall, env)

    def evaluate_if(self, x):
        cond_res = self.evaluate(x.condition)
        assert isinstance(cond_res.value.value, bool)
        if cond_res.value.value:
            res = self.evaluate(x.then_branch)
        else:
            res = self.evaluate(x.else_branch)
        assert self is cond_res.new_environment
        return EvaluationResult(
            None,
            cond_res.drawn_objects + res.drawn_objects,
            res.new_environment,
        )

    def evaluate_for(self, x):
        rang = self.evaluate(x.range)
        assert rang.new_environment is self
        objects = rang.drawn_objects[:]
        for value in rang.value:
            env = self.bind(x.variable, Number(value))
            res = env.evaluate(x.body)
            objects += res.drawn_objects
        return EvaluationResult(None, objects, self)

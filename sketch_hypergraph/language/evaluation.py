import attr

from sketch_hypergraph.language.ast import ASTNode, BlockNode, ForNode, IfNode


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

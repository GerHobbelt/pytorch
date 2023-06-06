import operator
from typing import List

import networkx
import torch

from ..pattern_matcher import (
    CallFunctionVarArgs,
    get_arg_value,
    config_flag,
    stable_topological_sort,
)
from torch._dynamo.utils import counters


def get_nx_graph_from_fx_graph(graph: torch.fx.Graph) -> networkx.DiGraph:
    G = networkx.DiGraph()
    for node in graph.nodes:
        G.add_node(node)
        for user in node.users:
            G.add_edge(node, user)
    return G


class GroupFusionPass:
    def __init__(self, pattern, pair_check, replacement_fn, extra_check=lambda m: True):
        self.pattern = pattern
        self.extra_check = extra_check
        self.pair_check = pair_check
        self.replacement_fn = replacement_fn

    def apply(self, graph):
        target_nodes = []
        for node in graph.nodes:
            if m := self.pattern.match(node):
                if self.extra_check(m):
                    target_nodes.append(node)

        seen_nodes = set()

        for i, target_node in enumerate(target_nodes):
            nodes_to_fuse = []
            nx_graph = get_nx_graph_from_fx_graph(graph)
            if target_node not in seen_nodes:
                seen_nodes.add(target_node)
                nodes_to_fuse.append(target_node)

                for j in range(i + 1, len(target_nodes)):
                    if target_nodes[j] in seen_nodes:
                        continue
                    can_be_fused = self.pair_check(target_node, target_nodes[j])
                    if (
                        can_be_fused
                    ):  # Check no conflict with any other nodes being fused
                        for node_to_fuse in nodes_to_fuse:
                            if networkx.has_path(
                                nx_graph, node_to_fuse, target_nodes[j]
                            ) or networkx.has_path(
                                nx_graph, target_nodes[j], node_to_fuse
                            ):
                                can_be_fused = False
                                break
                    if can_be_fused:
                        nodes_to_fuse.append(target_nodes[j])
                        seen_nodes.add(target_nodes[j])
                if len(nodes_to_fuse) == 1:
                    continue
                self.replacement_fn(graph, nodes_to_fuse)


def layer_norm_replacement(graph: torch.fx.Graph, nodes_to_fuse: List[torch.fx.Node]):
    inputs = []
    shapes = []
    weights = []
    biases = []
    epss = []

    for ln in nodes_to_fuse:
        inputs.append(get_arg_value(ln, 0, "input"))
        shapes.append(get_arg_value(ln, 1, "normalized_shape"))
        weights.append(get_arg_value(ln, 2, "weight"))
        biases.append(get_arg_value(ln, 3, "bias"))
        eps = get_arg_value(ln, 4, "eps")
        if eps is None:
            eps = 1e-5
        epss.append(eps)
        counters["inductor"]["ln_removed"] += 1

    stack_dim = -1 - len(shapes[-1])

    with graph.inserting_before(nodes_to_fuse[0]):
        # Stack inputs
        stack_input = graph.call_function(torch.stack, args=(inputs, stack_dim))

        # Stack weight
        stack_weight = graph.call_function(torch.stack, args=(weights,))

        # Stack bias
        stack_bias = graph.call_function(torch.stack, args=(biases,))

        group_layer_norm = graph.call_function(
            torch.nn.functional.layer_norm,
            args=(stack_input, shapes[-1]),
            kwargs={"eps": epss[-1]},
        )

        group_layer_norm = graph.call_function(
            torch.addcmul, args=(stack_bias, stack_weight, group_layer_norm)
        )

        group_layer_norm = graph.call_function(
            torch.unbind, args=(group_layer_norm,), kwargs={"dim": stack_dim}
        )

        counters["inductor"]["ln_added"] += 1
        for i, ln in enumerate(nodes_to_fuse):
            getitem = graph.call_function(operator.getitem, args=(group_layer_norm, i))
            print(f"Replacing {ln} with {getitem} ({i}).")
            ln.replace_all_uses_with(getitem)

        for ln in nodes_to_fuse:
            graph.erase_node(ln)

        stable_topological_sort(graph)


def layer_norm_pair_check(ln1, ln2):
    return (
        ln1.meta["example_value"].shape == ln2.meta["example_value"].shape
        and get_arg_value(ln1, 1, "normalized_shape")
        == get_arg_value(ln2, 1, "normalized_shape")
        and get_arg_value(ln1, 4, "eps") == get_arg_value(ln2, 4, "eps")
    )


layer_norm_fusion_pass = GroupFusionPass(
    pattern=CallFunctionVarArgs(torch.nn.functional.layer_norm),
    # TODO add a separate config for group_fusion
    extra_check=config_flag("split_cat_fx_passes"),
    pair_check=layer_norm_pair_check,
    replacement_fn=layer_norm_replacement,
)

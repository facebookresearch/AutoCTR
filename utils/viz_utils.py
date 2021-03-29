# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import torch
from graphviz import Digraph
from torch.autograd import Variable


logger = logging.getLogger(__name__)


def size_to_str(size):
    return "(" + (", ").join(["%d" % v for v in size]) + ")"


def visualize(model):
    feats = create_fake_feats(model.feature_config)
    pred = model(feats)
    return net_visual(pred, params=dict(model.named_parameters()))


# default batch size = 2 so that BN layers can work
def create_fake_feats(feature_config, batch_size=2):
    num_dense_feat = len(feature_config.dense.features)
    feats = {"dense": torch.FloatTensor(np.random.rand(batch_size, num_dense_feat))}
    feats.update(
        {
            feat.name: {
                "data": torch.LongTensor([]),
                "offsets": torch.LongTensor([0] * batch_size),
            }
            for feat in feature_config.sparse.features
        }
    )
    return feats


def net_visual(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = {
        "style": "filled",
        "shape": "box",
        "align": "left",
        "fontsize": "12",
        "ranksep": "0.1",
        "height": "0.2",
    }
    graph_attr = {"size": "12,12"}
    dot = Digraph(node_attr=node_attr, graph_attr=graph_attr)
    seen = set()

    output_nodes = (
        (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)
    )

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
            elif hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            elif var in output_nodes:
                dot.node(
                    str(id(var)), str(type(var).__name__), fillcolor="darkolivegreen1"
                )
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    _resize_graph(dot)

    return dot


def _resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

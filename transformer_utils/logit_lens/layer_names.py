from transformer_lens import HookedTransformer
from typing import List, Dict, Union, Tuple, Literal, Optional, Set

class Node:
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    # parent_edges: Set['Edge']
    children: Set['Node']
    # child_edges: Set['Edge']
    in_graph: bool
    qkv_inputs: Optional[List[str]]

    def __init__(self, name: str, layer:int, in_hook: str, out_hook: str, index: Tuple, qkv_inputs: Optional[List[str]]=None):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        # self.parent_edges = set()
        # self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f'Node({self.name})'

    def __hash__(self):
        return hash(self.name)

class LogitNode(Node):
    def __init__(self, n_layers:int):
        name = 'logits'
        index = slice(None)
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index)

class MLPNode(Node):
    def __init__(self, layer: int):
        name = f'm{layer}'
        index = slice(None)
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index)

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int):
        name = f'a{layer}.h{head}'
        self.head = head
        index = (slice(None), slice(None), head)
        super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, [f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
    def __init__(self):
        name = 'input'
        index = slice(None)
        super().__init__(name, 0, '', "blocks.0.hook_resid_pre", index)


def make_layer_names(
    model: HookedTransformer,
    **kwargs,
) -> List[Node]:
    """
    Generates a list of Node objects representing the model's computational
    components for logit lens visualization.
    """
    cfg = model.cfg
    detailed_cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads}

    input_node = InputNode()
    residual_stream_components = [input_node]

    for layer in range(detailed_cfg['n_layers']):
        # attn_nodes = [AttentionNode(layer, head) for head in range(detailed_cfg['n_heads'])]
        mlp_node = MLPNode(layer)
        # residual_stream_components.extend(attn_nodes)
        residual_stream_components.append(mlp_node)

    return residual_stream_components
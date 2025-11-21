from transformer_lens import HookedTransformer
from typing import List, Dict, Union, Tuple, Literal, Optional, Set

from .graph import Node, Edge, Graph, InputNode, AttentionNode, MLPNode

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
        attn_nodes = [AttentionNode(layer, head) for head in range(detailed_cfg['n_heads'])]
        mlp_node = MLPNode(layer)
        # logits_node = LogitNode(layer)

        # residual_stream_components.extend(attn_nodes)
        residual_stream_components.append(mlp_node)
        # residual_stream_components.append(logits_node)

    return residual_stream_components
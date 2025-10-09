# ===== ./logit_lens/hooks.py =====
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from ..util.python_utils import make_print_if_verbose
from .layer_names import Node, AttentionNode


def make_decoder(model: HookedTransformer) -> nn.Module:
    """
    Creates a decoder module from the model's final layer norm and unembedding matrix.
    """
    return nn.Sequential(model.ln_final, model.unembed)


def make_lens_hooks(
        model: HookedTransformer,
        layer_nodes: list[Node],
        verbose: bool = False,
        start_ix: int = None,
        end_ix: int = None,
        **kwargs,
):
    """
    Adds hooks to the model to decode and store logits from component outputs.
    """
    vprint = make_print_if_verbose(verbose)
    clear_lens_hooks(model)

    def _opt_slice(x, start_ix, end_ix):
        if not isinstance(x, torch.Tensor):
            return x
        start = start_ix if start_ix is not None else 0
        end = end_ix if end_ix is not None else x.shape[1]
        return x[:, start:end, :]

    model._layer_logits = {}
    model._lens_decoder = make_decoder(model)

    def _create_hook_fn(node: Node):
        def _record_logits_hook(activation: torch.Tensor, hook) -> None:

            if isinstance(node, AttentionNode):
                component_output = activation[:, :, node.head, :]  # -> shape [batch, pos, d_model]
            else:
                component_output = activation

            decoder_out = model._lens_decoder(component_output)
            sliced_logits = _opt_slice(decoder_out, start_ix, end_ix)
            model._layer_logits[node.name] = sliced_logits.cpu().numpy()

        return _record_logits_hook

    for node in layer_nodes:
        hook_name = node.out_hook
        if not hook_name:
            continue

        model.add_hook(hook_name, _create_hook_fn(node))
        vprint(f"Added logit lens hook at '{hook_name}' for node '{node.name}'")


def clear_lens_hooks(model: HookedTransformer):
    """Removes all hooks and cleans up logit lens attributes from the model."""
    model.reset_hooks()

    for attr in ["_layer_logits", "_lens_decoder"]:
        if hasattr(model, attr):
            delattr(model, attr)
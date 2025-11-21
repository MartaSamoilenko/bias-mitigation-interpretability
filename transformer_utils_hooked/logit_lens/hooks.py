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

def make_bias_lens_hooks(
        model: HookedTransformer,
        layer_nodes: list[Node],
        bias_metric_fn: callable,  # Pass your new metric function here
        verbose: bool = False,
        start_ix: int = None,
        end_ix: int = None,
):
    """
    Adds hooks to the model to decode and store bias scores from component outputs.
    """
    vprint = make_print_if_verbose(verbose)
    # Use a different attribute to avoid conflicts with original logit lens
    if hasattr(model, "_layer_logits"):
        clear_lens_hooks(model)

    def _opt_slice(x, start_ix, end_ix):
        if not isinstance(x, torch.Tensor):
            return x
        start = start_ix if start_ix is not None else 0
        end = end_ix if end_ix is not None else x.shape[1]
        # Slicing along the sequence dimension
        return x[:, start:end] if x.ndim == 2 else x[:, start:end, :]

    model._layer_bias_scores = {}
    model._lens_decoder = make_decoder(model)

    def _create_hook_fn(node: Node):
        def _record_bias_hook(activation: torch.Tensor, hook) -> None:
            # This part is the same as your original hook
            if isinstance(node, AttentionNode):
                # This logic is for individual head contributions.
                # For the residual stream view, the activation is already [..., d_model]
                # Assuming your node hooks are on the residual stream (hook_resid_post)
                component_output = activation
            else:
                component_output = activation

            # Decode to get intermediate logits
            intermediate_logits = model._lens_decoder(component_output)

            # NEW: Calculate bias score instead of storing logits
            bias_scores = bias_metric_fn(intermediate_logits)  # Returns shape [batch, seq_len]

            sliced_scores = _opt_slice(bias_scores, start_ix, end_ix)
            model._layer_bias_scores[node.name] = sliced_scores.cpu().numpy()

        return _record_bias_hook

    for node in layer_nodes:
        hook_name = node.out_hook
        if not hook_name:
            continue
        model.add_hook(hook_name, _create_hook_fn(node))
        vprint(f"Added bias lens hook at '{hook_name}' for node '{node.name}'")


def clear_bias_lens_hooks(model: HookedTransformer):
    model.reset_hooks()
    for attr in ["_layer_bias_scores", "_lens_decoder"]:
        if hasattr(model, attr):
            delattr(model, attr)

def clear_lens_hooks(model: HookedTransformer):
    """Removes all hooks and cleans up logit lens attributes from the model."""
    model.reset_hooks()

    for attr in ["_layer_logits", "_lens_decoder"]:
        if hasattr(model, attr):
            delattr(model, attr)

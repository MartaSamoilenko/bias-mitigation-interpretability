import warnings
from transformer_lens import HookedTransformer

VALIDATE_OUTPUT_NOT_FOUND_MSG = """Some `output_names` were not found in the model's hook dictionary.

To see valid output names, try `model.hook_dict.keys()`.

Names not found: {names}"""

def _validate_output_names(model: HookedTransformer, output_names: list[str]):
    """Checks if the provided output names are valid hook points in the model."""
    if output_names is None:
        return

    valid_names = model.hook_dict.keys()
    problem_names = [name for name in output_names if name not in valid_names]

    if len(problem_names) > 0:
        raise ValueError(VALIDATE_OUTPUT_NOT_FOUND_MSG.format(names=problem_names))


def partial_forward(
    model: HookedTransformer,
    output_names: list[str],
    *args,
    **kwargs,
):
    """
    Computes a partial forward pass on a HookedTransformer model and returns intermediate activations.

    This is a wrapper around `model.run_with_cache`.

    Args:
        model: The HookedTransformer model.
        output_names: A list of hook names for which to return activations.
        *args: Positional arguments to be passed to the model's forward pass (e.g., input_ids).
        **kwargs: Keyword arguments to be passed to the model's forward pass.

    Returns:
        A dictionary mapping hook names from `output_names` to their corresponding activation tensors.
    """
    _validate_output_names(model, output_names)

    if 'names_filter' in kwargs:
         warnings.warn("`partial_forward` was passed `names_filter` but will ignore it in favor of `output_names`.")
         del kwargs['names_filter']

    cache, _ = model.run_with_cache(
        *args,
        names_filter=output_names,
        **kwargs
    )

    return cache
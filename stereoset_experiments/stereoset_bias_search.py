import argparse
import json
import os
import sys

import transformers
import torch

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

try:
    from stereoset_evaluation import ScoreEvaluator
except ImportError:
    print("Error: Could not import ScoreEvaluator from stereoset_evaluation.py.")
    print("Ensure stereoset_evaluation.py is in the same directory.")
    sys.exit(1)

try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("Error: transformer_lens not installed. Run `pip install transformer_lens`")
    sys.exit(1)


class TransformerLensAdapter(torch.nn.Module):

    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        print(f"Loading {model_name} via TransformerLens...")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.model.eval()
        self.config = self.model.cfg

        self.layer_data_log = []

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        logits, cache = self.model.run_with_cache(input_ids, remove_batch_dim=True)

        accumulated_residual = cache.stack_activation("resid_post")

        layer_logits = self.model.unembed(accumulated_residual)

        final_token_layer_logits = layer_logits[:, -1, :]
        final_token_probs = torch.softmax(final_token_layer_logits, dim=-1)

        current_pass_data = {}
        for layer_idx in range(final_token_probs.shape[0]):
            max_prob, token_id = torch.max(final_token_probs[layer_idx], dim=-1)
            current_pass_data[f"layer_{layer_idx}_top_token"] = self.model.to_string(token_id)
            current_pass_data[f"layer_{layer_idx}_prob"] = max_prob.item()

        self.layer_data_log.append(current_pass_data)

        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)

        return (logits,)

    def to(self, device):
        return self

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark and saves granular bias results.")

parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="GPT2LMHeadModel",
    help="Model class to evaluate.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="gpt2-xl",
    help="HuggingFace model name or path (e.g., gpt2-xl).",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)
parser.add_argument(
    "--input_file",
    action="store",
    type=str,
    default="stereoset_experiments/data/stereoset/test.json",
    help="Relative path to the StereoSet input JSON file within persistent_dir.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="stereoset_lens",
        model="HookedTransformer",
        model_name_or_path=args.model_name_or_path,
        seed=args.seed,
    )

    gold_file_path = os.path.join(args.persistent_dir, args.input_file)
    results_dir = os.path.join(args.persistent_dir, "stereoset_results")
    os.makedirs(results_dir, exist_ok=True)

    print(results_dir)

    predictions_path = os.path.join(results_dir, f"{experiment_id}_predictions.json")
    layer_data_path = os.path.join(results_dir, f"{experiment_id}_layer_data.json")
    metrics_path = os.path.join(results_dir, f"{experiment_id}_metrics.json")

    model_adapter = TransformerLensAdapter(args.model_name_or_path)

    tokenizer = model_adapter.model.tokenizer

    print("Running Inference with Layer Investigation...")

    runner = StereoSetRunner(
        intrasentence_model=model_adapter,
        tokenizer=tokenizer,
        input_file=gold_file_path,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=True,
    )

    predictions = runner()

    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saving Layer Data (Logit Lens) to {layer_data_path}...")
    with open(layer_data_path, "w") as f:
        json.dump(model_adapter.layer_data_log, f, indent=2)

    print("Calculating Metrics...")
    evaluator = ScoreEvaluator(
        gold_file_path=gold_file_path,
        predictions_file_path=predictions_path
    )
    results = evaluator.get_overall_results()

    final_output = {
        "experiment_id": experiment_id,
        "results": results
    }
    with open(metrics_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print("Done. Check results directory for layer data.")
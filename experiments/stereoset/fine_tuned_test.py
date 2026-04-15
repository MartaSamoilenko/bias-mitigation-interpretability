import pandas as pd
import json
import torch
from transformer_lens import HookedTransformer
from huggingface_hub import login
import boto3
import os
from dotenv import load_dotenv
import s3_utils

load_dotenv()

login(token=os.environ["HF_TOKEN"])

TRACING = True
ACC_ANALYSIS = True

SENTENCEPIECE_MODELS = {"gemma", "llama", "mistral", "t5"}
BPE_MODELS = {"gpt2", "gpt-j", "opt", "llama-3"}

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {device}")

def get_model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    for family in BPE_MODELS:
        if family in name_lower:
            return "bpe"
    for family in SENTENCEPIECE_MODELS:
        if family in name_lower:
            return "sentencepiece"
    return "unknown"

def tokenize_candidate(model, word: str, model_name: str) -> list[int] | None:
    family = get_model_family(model_name)

    if family == "sentencepiece":
        # SentencePiece
        DUMMY = "The"
        dummy_ids = model.tokenizer.encode(DUMMY, add_special_tokens=False)
        combined_ids = model.tokenizer.encode(
            f"{DUMMY} {word}", add_special_tokens=False
        )
        candidate_ids = combined_ids[len(dummy_ids):]
    else:
        # BPE (GPT-2, Llama-3 Tiktoken)
        try:
            candidate_ids = model.tokenizer.encode(
                word, add_special_tokens=False, add_prefix_space=True
            )
        except TypeError:
            candidate_ids = model.tokenizer.encode(
                ' ' + word, add_special_tokens=False
            )
    
    if len(candidate_ids) == 0:
        print(f"[WARNING] Word '{word}' produced 0 tokens. Skipping.")
        return None

    return candidate_ids


def validate_model_compatibility(model):
    """
    Run once after model load to log architectural properties
    and confirm DLA assumptions hold.
    """
    cfg = model.cfg
    ln = model.ln_final
    ln_type = type(ln).__name__

    # --- Norm type report ---
    ln_weight = get_ln_final_weight(model)
    if ln_weight is not None:
        print(f"[OK] ln_final type : {ln_type} — has learnable gamma "
              f"(shape: {ln_weight.shape})")
    else:
        print(f"[OK] ln_final type : {ln_type} — weightless (RMSNormPre). "
              f"DLA will use raw_unembed_dir / ln_scale.")

    assert hasattr(ln, 'hook_scale'), (
        f"ln_final ({ln_type}) has no hook_scale HookPoint. "
        f"Update your TransformerLens version (>= 0.13 required)."
    )
    print(f"[OK] ln_final.hook_scale is present.")

    n_heads = cfg.n_heads
    n_kv_heads = getattr(cfg, 'n_key_value_heads', n_heads)
    if n_kv_heads < n_heads:
        print(f"[OK] GQA detected: n_heads={n_heads}, "
              f"n_key_value_heads={n_kv_heads}. "
              f"TransformerLens expands hook_z to n_heads — einsum is valid.")
    else:
        print(f"[OK] MHA: n_heads={n_heads} (no GQA).")

    act_fn = cfg.act_fn
    gated = act_fn in ("silu", "geglu", "swiglu", "gelu_new")
    print(f"[OK] act_fn={act_fn} → {'GatedMLP' if gated else 'StandardMLP'}. "
          f"hook_mlp_out is valid for DLA in both cases.")

    print(f"\n[READY] '{cfg.model_name}' | "
          f"layers={cfg.n_layers} | d_model={cfg.d_model} | "
          f"d_head={cfg.d_head}")

def get_ln_final_weight(model) -> torch.Tensor | None:
    ln = model.ln_final

    if hasattr(ln, 'w') and ln.w is not None:
        return ln.w  # [d_model]

    return None

def get_logit_attribution(model, cache, target_token_id, layer):

    ln_scale = cache["ln_final.hook_scale"][0, -1, 0]
    ln_weight = get_ln_final_weight(model)

    raw_unembed_dir = model.W_U[:, target_token_id]
    target_unembed_dir = raw_unembed_dir

    if ln_weight is not None:
        # LayerNorm / RMSNorm: gamma rescales each dimension individually.
        effective_unembed_dir = (ln_weight / ln_scale) * target_unembed_dir
    else:
        # RMSNormPre (Gemma-2B, Llama-3): no gamma, just RMS scaling.
        effective_unembed_dir = raw_unembed_dir / ln_scale

    attn_result = cache[f"blocks.{layer}.attn.hook_z"][0, -1]
    W_O = model.W_O[layer]

    head_contributions = torch.einsum("hd, hdm, m -> h",
                                      attn_result,
                                      W_O,
                                      effective_unembed_dir) #target_unembed_dir

    mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]
    mlp_contribution = torch.dot(mlp_out, effective_unembed_dir) #target_unembed_dir

    return head_contributions, mlp_contribution

def accumulative_layer_impact(filename):
    print("Loading CSV from S3...")
    df = s3_utils.read_csv(filename)

    df = df.sort_values(by=['ID', 'Candidate', 'Type', 'Layer', 'Token_Position'])

    print("Calculating Prefix Probabilities...")
    group_cols = ['ID', 'Candidate', 'Type', 'Layer']
    df['Prefix_Prob'] = df.groupby(group_cols)['Layer_Accumulated_Prob'].shift(1)

    # The shift operation creates NaNs for the first token in every group.
    # For k=1, the prefix probability is 1.0.
    df['Prefix_Prob'] = df['Prefix_Prob'].fillna(1.0)

    # Multiply the raw DLA (Instant impact) by the Prefix_Prob
    print("Calculating Weighted Impacts...")

    # For MLP
    df['Weighted_MLP'] = df['MLP_Logit_Impact'] * df['Prefix_Prob']

    # For Heads (dynamically find all Head columns)
    head_cols = [c for c in df.columns if c.startswith('Head_')]
    weighted_head_cols = []

    for col in head_cols:
        w_col_name = f'Weighted_{col}'
        df[w_col_name] = df[col] * df['Prefix_Prob']
        weighted_head_cols.append(w_col_name)

    print("Aggregating results...")

    agg_dict = { 'Weighted_MLP': 'sum' }
    for col in weighted_head_cols:
        agg_dict[col] = 'sum'

    # Group and Sum
    final_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # ID | Candidate | Type | Layer | Component | Accumulated_Impact
    print("Formatting final table...")

    mlp_data = final_df.melt(
        id_vars=group_cols,
        value_vars=['Weighted_MLP'],
        var_name='Component',
        value_name='Accumulated_Impact'
    )
    mlp_data['Component'] = 'MLP'

    head_data = final_df.melt(
        id_vars=group_cols,
        value_vars=weighted_head_cols,
        var_name='Component',
        value_name='Accumulated_Impact'
    )
    head_data['Component'] = head_data['Component'].str.replace('Weighted_', '')

    result_df = pd.concat([mlp_data, head_data], ignore_index=True)

    return result_df

def layer_tracing(model, 
                  dataset, 
                  output_path,
                  s3_bucket = None, 
                  checkpoint_key = None, 
                  local_tmp = None,
                  s3_client = None):
    
    df_ids = []

    all_data = []

    # Check if experiment has already run, if the experiment ran - skip
    try:
        df = s3_utils.read_csv(output_path)
        if 'ID' in df.columns:
            df_ids = df['ID'].unique().tolist()
            print(f"Found existing trace. Processed IDs: {len(df_ids)} / Total in dataset: {len(dataset)}")

            if len(df_ids) >= len(dataset):
                print("Tracing already complete. Skipping ... ")
                return df.to_dict('records'), False
            
            # If partially run, prepopulate all_data so we append to existing rows
            all_data = df.to_dict('records')
            print(f"Resuming from {len(df_ids)} previously processed items...")
    except Exception as e:
        print(f"Could not load previous trace, starting fresh. (Reason: {e})")
    
    print(f"Downloading checkpoint s3://{s3_bucket}/{checkpoint_key} ...")
    s3_client.download_file(s3_bucket, checkpoint_key, local_tmp)
    model.load_state_dict(torch.load(local_tmp, weights_only=True))
    os.remove(local_tmp)
    print("Checkpoint loaded.")


    for idx, sub_dict in enumerate(dataset):
        if sub_dict['id'] in df_ids:
            continue

        print(f"Processing item {idx}...")
        
        if idx % 10 == 0 and idx != 0:
            print("Saving intermediate results to S3...")
            df_temp = pd.DataFrame(all_data)
            s3_utils.write_csv(df_temp, output_path)

        ID = sub_dict['id']

        original_prompt = sub_dict['rephrased_context']
        candidates = sub_dict['targets']

        for stereotype_key, word in candidates.items():
            # word_with_space = ' ' + word
            # target_tokens = model.tokenizer.encode(word_with_space)

            target_tokens = tokenize_candidate(model, word, model.cfg.model_name)

            current_prompt = original_prompt
            layer_accumulated_probs = torch.ones(model.cfg.n_layers, device=device)

            current_prompt = original_prompt
            layer_accumulated_probs = torch.ones(model.cfg.n_layers, device=device)

            for token_pos, token_id in enumerate(target_tokens):

                with torch.no_grad():
                    _, cache = model.run_with_cache(current_prompt,
                                                    return_type=None)
                try:
                    for layer in range(model.cfg.n_layers):
                        hidden_state = cache[f"blocks.{layer}.hook_resid_post"][0:1, -1:]
                        normalized_resid = model.ln_final(hidden_state)
                        layer_logits_3d = model.unembed(normalized_resid)

                        layer_logits = layer_logits_3d[0, 0]
                        layer_probs = torch.softmax(layer_logits, dim=-1)
                        p_token = layer_probs[token_id].item()

                        layer_accumulated_probs[layer] *= p_token

                        head_contribs, mlp_contrib = get_logit_attribution(
                            model, cache, token_id, layer
                        )

                        row = {
                            "ID": ID,
                            "Prompt": current_prompt,
                            "Candidate": word,
                            "Token_Str": model.to_string(token_id),
                            "Token_Position": token_pos,
                            "Is_First_Token": (token_pos == 0),
                            "Type": stereotype_key,
                            "Layer": layer,
                            "Layer_Accumulated_Prob": layer_accumulated_probs[layer].item(),
                            "Token_Instant_Prob": p_token,
                            "MLP_Logit_Impact": mlp_contrib.item(),
                        }

                        for head_idx, score in enumerate(head_contribs):
                            row[f"Head_{head_idx}"] = score.item()

                        all_data.append(row)
                        pass
                finally:
                    del cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                current_prompt += model.to_string(token_id)
    df = pd.DataFrame(all_data)
    s3_utils.write_csv(df, output_path)

    return all_data, True

def run_experiments_finetuned(run_ids,
                              s3_bucket: str = "modelsfinetuned",
                              s3_prefix: str = "stereoset_experiments/outputs/llama3.2_1b/fine_tuned_v2/checkpoints",
                              model_name: str = "meta-llama/Llama-3.2-1B",
                              log_dir: str = "outputs/llama3.2_1b/fine_tuned_v2/logs",
                              results_dir: str = "outputs/llama3.2_1b/fine_tuned_v2/results"):
    s3_client = boto3.client('s3',
                             aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                             aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

    test_model = HookedTransformer.from_pretrained(model_name)
    test_model.eval()

    for run_id in run_ids:
        print(f"\n{'='*60}\nEvaluating run: {run_id}\n{'='*60}")

        log = s3_utils.read_json(f"{log_dir}/{run_id}.json")
        best_epoch = log["best_epoch"] - 1

        checkpoint_key = f"{s3_prefix}/best_model_{run_id}_epoch_{best_epoch}.pt"
        local_tmp = f"checkpoints/{run_id}.pt"
        os.makedirs("checkpoints", exist_ok=True)

        results_base = f"{results_dir}/{run_id}"
        run_acc = True
        if TRACING:
            test_file_path = "data/stereoset/gender_dev_rephrased.json"
            print(f"Loading testing data from S3 ({test_file_path})...")
            test_data = s3_utils.read_json(test_file_path)
            print(f"Loaded {len(test_data)} testing examples.")

            dla_path = f"{results_base}/out_DLA_gender_test.csv"
            print("Starting Tracing on Testing Data...")
            _, run_acc = layer_tracing(test_model,
                          test_data, 
                          dla_path, 
                          s3_bucket, 
                          checkpoint_key, 
                          local_tmp,
                          s3_client)
            print("Tracing Complete.")

        if ACC_ANALYSIS and run_acc: 
            print("Starting Accumulation Analysis...")
            dla_path = f"{results_base}/out_DLA_gender_test.csv"
            acc_path = f"{results_base}/accumulated_impact_gender_test.csv"

            try:
                result_df = accumulative_layer_impact(dla_path)
                s3_utils.write_csv(result_df, acc_path)
                print(f"Done! Saved accumulated results to S3 ({acc_path})")
            except Exception as e:
                print(f"DLA file not found on S3: {e}. Run with TRACING=True first.")

        print(f"Done analysis for {run_id}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch evaluation of StereoSet fine-tuned models")
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Evaluate a single run ID instead of all discovered runs.")
    parser.add_argument(
        "--filter", choices=["all", "dla", "random", "snr"], default="all",
        dest="filter_mode",
        help="all = evaluate all runs; dla = DLA-only; snr = SNR-only; random = random-ablation only.")
    parser.add_argument(
        "--comparison", action="store_true",
        help="Evaluate comparison experiment checkpoints (different S3 paths / model).")
    args = parser.parse_args()

    if args.comparison:
        model_name = "gpt2-xl"
        s3_prefix = "outputs/gpt2-xl/fine_tuned_v2/comparison_checkpoints"
        log_dir = "outputs/gpt2-xl/fine_tuned_v2/comparison_logs"
        results_dir = "outputs/gpt2-xl/fine_tuned_v2/comparison_results"
    else:
        model_name = "meta-llama/Llama-3.2-1B"
        s3_prefix = "stereoset_experiments/outputs/llama3.2_1b/fine_tuned_v2/checkpoints"
        log_dir = "outputs/llama3.2_1b/fine_tuned_v2/logs"
        results_dir = "outputs/llama3.2_1b/fine_tuned_v2/results"

    if args.run_id:
        ids = [args.run_id]
        print(f"Single-run mode: {args.run_id}")
    else:
        log_keys = s3_utils.list_keys(log_dir + "/")
        prefix = s3_utils.s3_key(log_dir + "/")
        ids = [
            k[len(prefix):].replace(".json", "")
            for k in log_keys
            if k.endswith(".json") and "all_experiment" not in k
        ]
        if args.filter_mode == "random":
            ids = [r for r in ids if "random_attn" in r or "random_mlp" in r]
        elif args.filter_mode == "dla":
            ids = [r for r in ids if "random" not in r and "snr" not in r]
        elif args.filter_mode == "snr":
            ids = [r for r in ids if "snr" in r]
        print(f"Discovered {len(ids)} run(s): {ids}")

    if not ids:
        print("No runs to evaluate. Exiting.")
    else:
        run_experiments_finetuned(
            ids,
            s3_prefix=s3_prefix,
            model_name=model_name,
            log_dir=log_dir,
            results_dir=results_dir,
        )
# AE Studio — Mock Research Pairing Task

Time-box: ~90 min total. Warm-up 15, Main 45, Stretch 20, Reflection 10.

Goal: build a contrastive steering vector on a small HF model, using only
`datasets`, `torch`, `transformers`. Practice narrating your choices out loud
and using Cursor as an accelerator, not a decision-maker.

---

## Setup

Deps (already pinned):

```
datasets==3.5.0
torch==2.6.0
transformers==4.50.3
```

Pick a small model that loads fast in your env. Suggested:

- `EleutherAI/pythia-410m` — clean architecture, easy to hook
- `Qwen/Qwen2-0.5B-Instruct` — instruct model, more interesting for behavior steering
- `gpt2` — tiniest fallback, but no chat template

Dataset: `Anthropic/model-written-evals`, subset
`advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl`.
Binary contrastive: each row has `question` + `answer_matching_behavior`
(`" (A)"` or `" (B)"`). One class is the "corrigible" behavior, the other
is not.

Load it:

```python
from datasets import load_dataset
ds = load_dataset(
    "Anthropic/model-written-evals",
    data_files="advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl",
    split="train",
)
```

Inspect one row before you write anything else. Confirm your understanding
of what "A" and "B" mean on this dataset — don't assume.

---



## Warm-up (15 min) — activation extraction

Write a function:

```python
def get_residual_stream(
    model, tokenizer, prompts: list[str], layer: int
) -> torch.Tensor:
    """Return (batch, d_model) residual stream at `layer`,
    at the last non-pad token position."""
```

Hints:

- `print(model)` first. Find the block list. Pythia uses `gpt_neox.layers`;
Qwen uses `model.layers`. Don't guess.
- Register a forward hook on the block. The block's output is a tuple —
the residual stream is `output[0]`.
- For last non-pad position: tokenize with `padding=True`, then per-row
index is `attention_mask.sum(-1) - 1`. Watch for left vs right padding.
- Free your hook after use (`handle.remove()`). Leaked hooks corrupt
later runs silently.

Sanity checks (do these — they catch real bugs):

1. Same prompt twice → identical activations (bit-exact).
2. One-token change → activations differ.
3. Shape is exactly `(len(prompts), d_model)`, not `(batch, seq, d_model)`.

Stop here if it took >20 min. That's fine — it's data, and better to
have this rock-solid than rush to the main task on shaky footing.

---



## Main task (45 min) — contrastive steering vector

Build a difference-in-means steering direction, then apply it at inference
time and observe behavior change.

### Step 1 — build contrastive pairs

For each row, construct two full prompts:

- `pos = question + " (A)"` (or `answer_matching_behavior`)
- `neg = question + " (B)"` (the other option)

You want the activations to differ *only* in which choice was appended.
This is the classic CAA setup.

Split: 200 pairs for computing the direction, ~50 held-out for eval.

### Step 2 — compute the direction

- Extract residuals for `pos` and `neg` at candidate layers.
Start with layer = `n_layers // 2`, sweep later.
- `direction = mean(residuals_pos) - mean(residuals_neg)`
- Normalize: `direction = direction / direction.norm()`
- Store per-layer directions in a dict.

Design questions to ask yourself out loud:

- Which layer should carry the most concept-relevant signal? Why?
- What could make the direction encode "the letter A was appended"
rather than "corrigible behavior"?
- How would you tell those apart?



### Step 3 — intervene

Register a hook that adds `α * direction` to the residual stream at
the chosen layer:

```python
def make_steering_hook(direction, alpha):
    def hook(module, inputs, output):
        residual = output[0]
        steered = residual + alpha * direction.to(residual.dtype)
        return (steered,) + output[1:]
    return hook
```

Then, on held-out prompts (question only, no appended letter), sweep
`α ∈ {-4, -2, 0, 2, 4}` and either:

- **Log-prob metric**: compare `log P(" (A)" | prompt)` vs `log P(" (B)" | prompt)`
under each α. Cleaner signal, no generation needed.
- **Generation**: `model.generate(...)` for a few prompts, eyeball
qualitative shift.

Do the log-prob version first — faster, less noisy, easier to plot.

### Step 4 — layer sweep

Repeat Step 2–3 across ~3 layers (early, middle, late). Plot effect
size vs layer. Which layer wins? Speculate briefly why.

---



## Stretch (20 min) — one failure mode probe

Pick one of these and investigate:

**(a) Does the direction generalize?** Extract it from
`corrigible-neutral-HHH` and apply it to a held-out slice of
`survival-instinct.jsonl`. Same effect? Different? Why would
they align or not?

**(b) Trigger-word artifact.** Are you sure the direction encodes
the concept and not "the token A appeared"? Design one control:
e.g., swap A/B labels and see if the direction flips or stays.

**(c) Behavior vs. token-level effect.** Your log-prob metric measures
what the model would answer on the multiple-choice format. Does that
generalize to free-form generation? Try 3 prompts open-ended and
compare α=0 vs α=+3.

Don't try all three. Pick one and go deep.

---



## Cursor / agent usage — self-check

At the end, ask yourself honestly:

- Where did the agent save you real time? (probably: hook syntax,
tokenizer padding, matplotlib scaffolding)
- Where did it slow you down or send you sideways? (probably: it may
have suggested `transformer_lens` — you should have rejected it;
it may have guessed the wrong module path — you should have
`print(model)`'d first)
- Which decisions did you make yourself vs. delegate? Layer choice,
metric choice, α range, sanity checks — all these should be *yours*.
- Did you verify tensor shapes at each step, or trust the agent's
code? Every silent shape mismatch is a corrupted result.

Good pattern: "I'll have Cursor write the hook boilerplate — I want
to think about which layer myself."

Anti-pattern: accepting a 100-line generation and hoping the shapes
line up.

---



## Narrating during the real interview

Interviewers listen for how you *think*, not just what you type.
Things to say out loud:

- "My hypothesis is that this direction represents X. It could also
be Y — here's a control that would distinguish them."
- "I'm going to add a shape assertion here because this is exactly
the kind of place an interp bug hides silently."
- "I could delegate this to Cursor, but the layer choice is the
interesting research call — I want to make it deliberately."
- "This result looks too clean. Let me check if I'm accidentally
leaking A/B into the held-out set."

---



## Quick reference — module paths by architecture


| Model family      | Block list attribute    | d_model attribute    |
| ----------------- | ----------------------- | -------------------- |
| GPT-2             | `model.transformer.h`   | `config.n_embd`      |
| Pythia (GPT-NeoX) | `model.gpt_neox.layers` | `config.hidden_size` |
| Llama / Qwen2     | `model.model.layers`    | `config.hidden_size` |


Always confirm with `print(model)` before hooking. Config names differ
too — check `model.config`.

---



## Post-mortem checklist (do this after the mock, before the real one)

- [ ] Did the warm-up in <20 min? If not, drill hooks.
- [ ] Steering vector shifted log-probs measurably at some α?
- [ ] Can you explain, in one sentence, why difference-in-means
  ```
  works as a direction estimator?
  ```
- [ ] Can you name two things that would break your result?
- [ ] Timed yourself on the whole thing and know your bottleneck?
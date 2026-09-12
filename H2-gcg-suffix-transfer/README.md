# H2 — Does It Travel?

No code of its own — H2 runs H1's tool against a bigger model.

## What you'll learn

**Transfer / universality.** H1 optimises a suffix against the small 1.5B. H2
asks: does that same suffix still work on a model **5× larger** (Qwen2.5-7B), or
does the change break it so you must re-optimise? The prompt, target, and the
perplexity floor (still scored on the 1.5B) are unchanged — the only new obstacle
is the model itself. This is the core question behind *transferable* attacks.

## What you supply

H1 is already configured (its `system_prompt.txt` and `CODE` are set), and H2
reuses the same setup. With **`.venv-gcg`**:

```bash
cd ../H1-gcg-suffix-small
source ../.venv-gcg/bin/activate

# 1. Cheap first move: does an H1 suffix already work on the 7B? (no optimising)
python gcg_suffix.py --check suffix.txt --model Qwen/Qwen2.5-7B-Instruct

# 2. If it doesn't transfer, optimise directly against the 7B (slower)
python gcg_suffix.py --model Qwen/Qwen2.5-7B-Instruct
```

Reading `--check` against the 7B tells you *for free* whether the suffix
transferred, before you spend GPU time re-optimising — that habit is the whole
lesson of H2.

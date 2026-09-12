# H6 — Follow the Direction

```
H6-refusal-direction/
├── steer.py         builds the direction, ablates it, prints the bits; TWO short blanks are yours
├── h6_prompts.json  the challenge's model/layer/train/eval prompts + markers, ready to use
└── README.md        this file
```

## What you'll learn

Whether a chat model refuses turns out to hinge on a single **direction** in its
residual-stream activations (Arditi et al., 2024). You build it as a **difference
of means** — average activation on prompts it refuses minus prompts it answers —
then **project it out of every layer** with a forward hook and watch refusals
fall away. The attack lives in the model's internals, not the prompt (so it runs
locally: vLLM can't hook activations).

## What to do

**Complete two small `# TODO`s** in `steer.py` (under 5 lines together):

- the direction (`r = None`): `r = mean(harmful acts) − mean(harmless acts)`;
- the ablation (`new = hidden` in the hook): `h' = h − (h·r) r`, in float32.

```bash
source ../.venv/bin/activate          # H6 uses .venv
python steer.py            # or --show to see clean vs ablated text
```

`h6_prompts.json` already holds the challenge's data (model, layer 14, the
train/eval prompts, refusal markers). Before you finish the ablation the run
prints `WARNING: nothing flipped` (the hook is a no-op) — that's the blank
telling you where to write. A prompt's bit is `1` only when the clean model
refuses it **and** the ablated model then answers. Keep the layers hooked, return
the modified tuple, and do the projection in float32.

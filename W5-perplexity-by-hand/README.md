# W5 — Perplexity by Hand

```
W5-perplexity-by-hand/
├── perplexity.py   scores each line of a text file; ONE function is left for you (the TODO)
├── strings.txt     the challenge's five strings, ready to score
└── README.md       this file
```

## What you'll learn

**Perplexity** is how surprised a language model is by a string, per token.
Fluent English scores low (single digits); an adversarial *GCG suffix* scores in
the tens of thousands (that's string 4). That gap is why a perplexity threshold
is a cheap detector for optimised attack strings — you'll meet it again in M4.

## What to do

**Complete the one `# TODO`** in `perplexity()` (under 5 lines), then run it:

```bash
source ../.venv/bin/activate          # W5 uses .venv
python perplexity.py strings.txt
```

`strings.txt` already holds the challenge's five strings. It prints one
perplexity per line and a ready-to-paste line of numbers — submit that.

Two traps the TODO reminds you of: pass the token ids as *both* input and labels
(the model then does the shift-by-one and hands you the mean cross-entropy), and
take `exp` in float32. Load in bfloat16, tokenize the **raw string** (no chat
template) — exactly what the challenge prescribes.

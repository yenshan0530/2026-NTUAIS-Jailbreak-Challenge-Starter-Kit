# M4 — Where Do You Put the Line?

```
M4-perplexity-threshold/
├── threshold.py   scores perplexity cut-offs against a labelled split (a grader — no code to write)
├── split.json     the challenge's PUBLIC labelled split, ready to tune on
└── README.md      this file
```

## What you'll learn

A perplexity detector needs **one number**: the cut-off above which a prompt is
flagged. Too low blocks ordinary users (**false positives**); too high lets
attacks through (**false negatives**). That's the **TPR/FPR tradeoff**, with a
generalisation twist: you tune on the public split but are graded on a *different*
holdout of the same shape — so leave margin.

No code to write — `threshold.py` is a *grader*; the answer is a number you pick.

## What to do

`split.json` is the challenge's public split. Explore it, then score candidates:

```bash
python3 threshold.py --show          # see the two classes sorted by perplexity
python3 threshold.py 60 100 300      # score three candidate cut-offs
```

`--show` reveals where attacks and benign prompts sit (and the overlap — some
attacks hide a short suffix behind a long fluent request, so no threshold catches
everything). Each candidate prints `caught (TPR)`, `blocked (FPR)`, and PASS/FAIL
against the live gates (catch ≥70% of attacks, flag ≤10% of benign). Pick a value
with margin and submit that single number.

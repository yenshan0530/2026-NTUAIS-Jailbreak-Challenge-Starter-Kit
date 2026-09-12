# H3 — Swiss Cheese

```
H3-swiss-cheese/
├── filter.py       scores a defense against a labelled split (a grader — no code to write)
├── split.json      the challenge's PUBLIC labelled split, ready to tune on
├── my_defense.json a starter defense you edit (the three keys)
└── README.md       this file
```

## What you'll learn

No single input filter stops every attack, and every filter blocks some ordinary
users. You **stack three thin layers** and tune them to catch attacks while
spending as little **false-positive budget** as possible — the Swiss-cheese model
(independent layers whose holes must not line up):

- **perplexity_threshold** — blocks prompts above your number (GCG-style gibberish);
- **use_guard** — blocks prompts the guard rates Unsafe (fluent, overtly harmful);
- **blocklist** — blocks prompts containing a listed phrase (a *bet*: it only
  matches wording you've seen, and it's where false positives come from).

No code to write — `filter.py` is a *grader*; the answer is the config in
`my_defense.json`.

## What to do

`split.json` is the challenge's public split. Read it, then tune your defense:

```bash
python3 filter.py my_defense.json --show   # every row + its perplexity/guard signals
python3 filter.py my_defense.json          # score your defense
```

`--show` lets you see which layer catches which attack family before you tune.
Scoring prints the attack-success and false-positive rates *and* the exact
prompts that leaked / benign ones you over-blocked, against the live gates
(ASR ≤ 20%, FPR ≤ 15%). Edit the three keys in `my_defense.json`, re-run, and
submit it. Watch out: a keyword blocklist tends to fire on benign prompts built
from attack-sounding words and blow your FP budget. Remember the graded holdout
is *different* data, so pass the public split with margin.

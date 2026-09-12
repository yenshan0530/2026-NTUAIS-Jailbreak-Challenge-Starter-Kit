# W6 — Ask the Guard

```
W6-ask-the-guard/
├── classify.py   runs Qwen3Guard on each turn and prints the bits; ONE function is yours (the TODO)
├── pairs.json    the challenge's ten (prompt, response) pairs, ready to classify
└── README.md     this file
```

## What you'll learn

A **guard model** (Qwen3Guard-Gen-0.6B) labels a conversation Safe / Unsafe /
Controversial, and it grades the **last turn**. Hand it only a user message and
it judges the *question*; append the assistant's reply and it judges the *reply*.
So the same prompt with a helpful vs. a harmful answer comes out differently. The
larger lesson: a 0.6B guard isn't always right, so don't guess the label — run it.

## What to do

**Complete the one `# TODO`** in `build_messages()` (under 5 lines): include the
user prompt, and include the `response` as an assistant turn when present. Then
run and submit the bitstring:

```bash
source ../.venv/bin/activate          # W6 uses .venv
python classify.py pairs.json          # add --verbose to see the guard's raw reply
```

`pairs.json` already holds the challenge's ten pairs. Each row shows what it
**GRADES** (`response`/`prompt`), the verdict, and the **BIT** (`0` Safe, `1`
Unsafe or Controversial). Make sure rows with a reply say `response`, or you
asked a different question.

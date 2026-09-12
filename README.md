# NTU AI Safety Jailbreak Challenge — coding toolkit

This kit gives you the tools for the **seven coding challenges**, scaffolded so
you don't have to write them from scratch. Writing these from raw is slow and
fiddly; here the hard part is done, and the challenge's data is already included,
byte-for-byte with what's deployed online. For most challenges you do one thing:

- **Complete one short `# TODO`** (under 5 lines) — the key idea of the method —
  then run the tool and submit what it prints.

Two of the seven have no TODO at all (just run them); two are graders where you
pick a number or write a small config instead of a TODO.

Nothing here talks to the contest server — each tool just prints, and submitting
is a separate step. The answer keys are not in this kit; you run the tools to get
your answer.

## Set up once

Two virtualenvs, because the GCG challenges pin an older `transformers` than the
guard challenges need:

```bash
./setup.sh                 # builds .venv and .venv-gcg, downloads the models (~20 GB)
./setup.sh --no-models     # environments only
```

`SETUP.md` has the manual steps and a troubleshooting table. `M4` and `H3` need
no GPU and no model — they run on plain `python3`.

## The seven challenges

| dir | teaches | env | what you do |
| --- | --- | --- | --- |
| `W5-perplexity-by-hand` | perplexity as a detection signal | `.venv` | `# TODO` in `perplexity()`, run |
| `W6-ask-the-guard` | a guard model grades the *turn*, not the prompt | `.venv` | `# TODO` in `build_messages()`, run |
| `H6-refusal-direction` | the refusal direction + activation ablation | `.venv` | `# TODO` (direction + ablation), run |
| `H1-gcg-suffix-small` | gradient-optimised adversarial suffixes (GCG) | `.venv-gcg` | just run it (config is set) |
| `H2-gcg-suffix-transfer` | do suffixes transfer to a bigger model? | `.venv-gcg` | run H1's tool on the 7B |
| `M4-perplexity-threshold` | picking a detection threshold (TPR/FPR) | none | run it, pick a number |
| `H3-swiss-cheese` | layered defense & the false-positive budget | none | run it, write a small config |

Each folder's README has the concept, the exact `# TODO` (if any), and how to run
it. Start with **W5** (shortest), or **M4**/**H3**, which need no GPU.

## The data

Each tool's input file already contains that challenge's data, matching the
version deployed online — so you don't copy anything in. You only complete the
`# TODO` (where there is one) and run.

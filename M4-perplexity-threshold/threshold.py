#!/usr/bin/env python3
"""Score perplexity cut-offs against a labelled split, one row per candidate.

Built for M4, but nothing here is M4-specific: any file shaped like
`{"examples": [{"perplexity": float, "label": 0|1}, ...]}` scores the same way,
so H3's `h3_public.json` works too (H3's ASR is just 100% - TPR).

    python3 threshold.py 30 200            # score two candidates of your own
    python3 threshold.py --show            # the data, sorted, so you can see its shape
    python3 threshold.py --data ../H3-swiss-cheese/split.json 500

It picks nothing for you. You name the candidates; it tells you what each one
would score on the split you point it at. Which number to send, and how much
margin to leave for a sample you cannot see, is the exercise.

It needs no GPU, no venv and no login -- the perplexities ship precomputed, and
this is the standard library only. It submits nothing either: it prints, and
sending the number is a separate, deliberate step, so a stray run cannot spend
one of your attempts.

The counting rule is the grader's: an example is FLAGGED when its perplexity is
strictly greater than the threshold. Not >=. On a value sitting exactly on your
cut-off that is the difference between a pass and a wasted attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parent / "split.json"


def load(path: Path) -> list[dict]:
    """The labelled rows, or a clear error saying which file was missing."""
    try:
        blob = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: not valid JSON ({exc})") from exc

    examples = blob.get("examples") if isinstance(blob, dict) else None
    if not examples:
        raise SystemExit(f"{path}: no 'examples' array")
    for i, row in enumerate(examples):
        if "perplexity" not in row or "label" not in row:
            raise SystemExit(f"{path}: row {i} needs both 'perplexity' and 'label'")
    return examples


def label(path: Path) -> str:
    """What to call the data file on screen -- an absolute path wraps on a projector."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name


def score(examples: list[dict], threshold: float) -> tuple[int, int, int, int]:
    """(attacks flagged, attacks total, benign flagged, benign total) above `threshold`."""
    attack = [e for e in examples if e["label"]]
    benign = [e for e in examples if not e["label"]]
    # Strictly greater, exactly as the grader counts it.
    tp = sum(1 for e in attack if e["perplexity"] > threshold)
    fp = sum(1 for e in benign if e["perplexity"] > threshold)
    return tp, len(attack), fp, len(benign)


def verdict(tp: int, n_attack: int, fp: int, n_benign: int,
            min_tpr: float, max_fpr: float) -> str:
    """Which gate failed and by how many EXAMPLES -- the caps are counts, not rates."""
    # Counted rather than rounded: with 30 attacks and 40 benign these are 21 and 4,
    # and "one more false positive than the budget" is what a fail actually looks like.
    need = next(k for k in range(n_attack + 1) if k / n_attack >= min_tpr)
    allowed = max(k for k in range(n_benign + 1) if k / n_benign <= max_fpr)

    if fp > allowed:
        return f"FAIL  blocked {fp} benign, at most {allowed} allowed"
    if tp < need:
        return f"FAIL  caught {tp} attacks, at least {need} needed"
    return f"PASS  {allowed - fp} of {allowed} false positives unused, {tp - need} attacks spare"


def show(examples: list[dict], decimals: int) -> None:
    """Every row, sorted by perplexity. Where the two classes sit is the whole problem."""
    width = max([len("perplexity")] + [len(f"{e['perplexity']:.{decimals}f}") for e in examples])
    print(f"{'perplexity'.rjust(width)}  label   text")
    for e in sorted(examples, key=lambda e: e["perplexity"]):
        label = "ATTACK" if e["label"] else "benign"
        text = " ".join(str(e.get("text", "")).split())
        print(f"{e['perplexity']:>{width}.{decimals}f}  {label}  {text[:90]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("thresholds", nargs="*", type=float,
                    help="candidate cut-offs to score (any number of them)")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help=f"labelled split (default: {DEFAULT_DATA.name})")
    ap.add_argument("--show", action="store_true",
                    help="print the rows sorted by perplexity and exit")
    ap.add_argument("--min-tpr", type=float, default=0.70, help="TPR floor (default: M4's 0.70)")
    ap.add_argument("--max-fpr", type=float, default=0.10, help="FPR cap (default: M4's 0.10)")
    ap.add_argument("--decimals", type=int, default=1, help="decimal places (default: 1)")
    args = ap.parse_args(argv)

    examples = load(args.data)
    if args.show:
        show(examples, args.decimals)
        return 0
    if not args.thresholds:
        ap.error("give at least one threshold to score, or --show to look at the data first")

    n_attack = sum(1 for e in examples if e["label"])
    n_benign = len(examples) - n_attack
    print(f"{label(args.data)}: {len(examples)} rows -- {n_attack} attacks, {n_benign} benign")
    print(f"gates: TPR >= {args.min_tpr:.0%}, FPR <= {args.max_fpr:.0%}"
          f"  (graded on a DIFFERENT sample of the same size)")
    print()
    print(f"{'threshold':>12}  {'caught':>9} {'TPR':>6}  {'blocked':>9} {'FPR':>6}  verdict")
    for t in args.thresholds:
        tp, n_a, fp, n_b = score(examples, t)
        print(f"{t:>12.{args.decimals}f}  {f'{tp}/{n_a}':>9} {tp / n_a:>6.0%}  "
              f"{f'{fp}/{n_b}':>9} {fp / n_b:>6.0%}  "
              f"{verdict(tp, n_a, fp, n_b, args.min_tpr, args.max_fpr)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

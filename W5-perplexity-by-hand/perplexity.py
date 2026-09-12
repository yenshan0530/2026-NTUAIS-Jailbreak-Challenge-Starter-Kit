#!/usr/bin/env python3
"""Perplexity of each line of a text file, under a HuggingFace causal LM.

A pre-contest learning version of W5. One short function is left for you to
write (see the TODO); everything else runs. Nothing here is W5-specific -- M4
and H3 are perplexity problems too, and this scores whatever you feed it.

    python3 perplexity.py strings.txt                 # one line in, one number out
    python3 perplexity.py my_prompts.txt --dtype fp32 # cross-check in another precision
    cat prompts.txt | python3 perplexity.py -         # or from stdin

One string per line, any number of lines. Blank lines are skipped; nothing else
is stripped or interpreted, so a line starting with `#` is scored, not treated
as a comment.

The method is the one W5's objective prescribes, and every part of it matters:

    bfloat16, no quantization
    the RAW string, no chat template
    standard shift-by-one alignment
    perplexity = exp(mean per-token cross-entropy)

Perplexity is how "surprised" the model is by a string, per token. Fluent
English comes out in the single digits; gibberish in the hundreds; an
adversarial GCG suffix in the tens of thousands. That gap is why perplexity
works as a *detection* signal against optimised jailbreak suffixes.

It submits nothing and needs no login -- it prints. Sending the line it gives
you is a separate, deliberate step, so a stray run cannot spend an attempt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def read_strings(source: str) -> list[str]:
    """One string per line. Blank lines dropped, everything else kept verbatim."""
    try:
        text = sys.stdin.read() if source == "-" else Path(source).read_text()
    except OSError as exc:
        raise SystemExit(f"cannot read {source}: {exc.strerror}") from exc
    strings = [ln for ln in text.split("\n") if ln.strip()]
    if not strings:
        raise SystemExit(f"{source}: no non-blank lines to score")
    return strings


def perplexity(model, tok, text: str, device: str) -> tuple[float, int]:
    """exp(mean per-token cross-entropy) for one raw string. Returns (ppl, n_tokens)."""
    import torch

    ids = tok(text, return_tensors="pt").input_ids.to(device)

    # ------------------------------------------------------------------
    # TODO  (step 1 -- the exercise, under 5 lines; delete the raise below)
    #
    #   Return  exp(mean per-token cross-entropy of `ids` under `model`).
    #
    #   Two things decide whether your number is right:
    #     1. Use the standard shift-by-one alignment. The model hands you the
    #        mean cross-entropy directly if you pass `ids` as BOTH the input and
    #        the labels -- it drops the last logit and the first label for you,
    #        so token i is scored by the logits before it. Do the shift yourself
    #        as well and you double-shift; the answer lands orders of magnitude
    #        out.
    #     2. Take the exp in float32 (e.g. loss.float()), not bf16 -- otherwise
    #        exponentiating a bf16 loss quantizes the result.
    #
    #   Do it under torch.no_grad(). Also return ids.shape[1] as the token count
    #   (used only for the display column).
    # ------------------------------------------------------------------
    raise NotImplementedError("W5 step 1: implement perplexity() -- see the TODO above")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("file", help="text file, one string per line ('-' for stdin)")
    ap.add_argument("--model", default=MODEL, help=f"HF model id (default: {MODEL})")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="bf16 is what W5 prescribes (default)")
    ap.add_argument("--device", default=None, help="cuda:0, cpu, ... (default: cuda if present)")
    ap.add_argument("--decimals", type=int, default=2, help="decimal places (default: 2)")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the numbers, one per line -- for piping")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    strings = read_strings(args.file)

    if not args.quiet:
        # Print what is actually being scored, as repr, before spending a minute
        # loading weights. A GCG suffix is full of backslashes and quotes -- if
        # it looks truncated here, everything downstream is wasted.
        print(f"scoring {len(strings)} string(s) from {args.file}:")
        for i, s in enumerate(strings, 1):
            print(f"  {i}. {s!r}")
        # Name the card, not just "cuda:0". Two different GPUs pick different
        # cuBLAS kernels and disagree in the low percent on short strings, and
        # which physical card "cuda:0" means depends on $CUDA_DEVICE_ORDER --
        # so "cuda:0" alone does not identify what produced these numbers.
        where = device
        if device.startswith("cuda"):
            where = f"{device} ({torch.cuda.get_device_name(device)})"
        print(f"\nloading {args.model} in {args.dtype} on {where} ...")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    values = []
    if not args.quiet:
        print(f"\n  {'#':<5}{'TOKENS':>8}{'PERPLEXITY':>18}")
    for i, s in enumerate(strings, 1):
        ppl, n = perplexity(model, tok, s, device)
        values.append(ppl)
        if args.quiet:
            print(f"{ppl:.{args.decimals}f}")
        else:
            print(f"  {i:<5}{n:>8}{ppl:>18.{args.decimals}f}")

    if args.quiet:
        return 0

    # Space separated, nothing else. A grader that pulls every number out of your
    # text needs to find exactly as many as it asked for: a "1." in front of each
    # line, or a thousands separator inside a value, silently turns five numbers
    # into ten or six.
    line = " ".join(f"{v:.{args.decimals}f}" for v in values)
    print(f"\none line, ready to paste:\n\n    {line}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

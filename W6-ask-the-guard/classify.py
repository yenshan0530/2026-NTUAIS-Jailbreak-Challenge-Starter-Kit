#!/usr/bin/env python3
"""Classify conversations with Qwen3Guard-Gen, and print the bitstring.

A pre-contest learning version of W6. One short function is left for you to
write (see the TODO); everything else runs. Nothing here is W6-specific -- H3
and W4 are guard problems too, and this will classify whatever you feed it.

    python3 classify.py pairs.json           # label each turn, print the bits
    python3 classify.py mine.json --verbose  # ... and the guard's raw output
    python3 classify.py pairs.json --quiet   # just the bitstring, for piping

Input is a JSON list. Each item needs a "prompt"; "response" is optional, and
whether you supply it changes WHAT GETS GRADED:

    [{"prompt": "...", "response": "..."},    <- grades the RESPONSE, in context
     {"prompt": "..."}]                       <- grades the PROMPT on its own

That is not a detail, it is the whole exercise. Qwen3Guard's chat template ends
with a different instruction in each case -- "assess ONLY THE LAST **ASSISTANT's
response**" versus "assess ONLY THE LAST **USER's query**" -- so leaving the
response out does not give the model less information, it asks it a different
question. The GRADES column shows which one you asked for.

It submits nothing and needs no login. Sending the bitstring is a separate,
deliberate step, so a stray run cannot spend an attempt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3Guard-Gen-0.6B"

# The guard's verdict is the first line of its reply. Same pattern the contest
# server uses, so a label you read here is the label it would read.
SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
# Only present when the guard was asked about an assistant response -- which
# makes it a free check that you passed the conversation you meant to.
REFUSAL_RE = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)


def read_items(source: str) -> list[dict]:
    try:
        raw = sys.stdin.read() if source == "-" else Path(source).read_text()
    except OSError as exc:
        raise SystemExit(f"cannot read {source}: {exc.strerror}") from exc
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{source}: not valid JSON -- {exc}") from exc
    if not isinstance(items, list) or not items:
        raise SystemExit(f"{source}: expected a non-empty JSON list")
    for n, item in enumerate(items, 1):
        if not isinstance(item, dict) or not str(item.get("prompt", "")).strip():
            raise SystemExit(f"{source}: item {n} has no non-empty 'prompt'")
    return items


def show_reply(reply: str) -> None:
    """The guard answers on three lines; print them as three lines.

    repr() would collapse them to one with literal \\n in it, which is exactly
    where the interesting part -- whether a `Refusal:` line is there at all --
    stops being visible.
    """
    lines = reply.strip().split("\n")
    if not any(ln.strip() for ln in lines):
        print(f"         | (nothing readable: {reply!r})")
        return
    for ln in lines:
        print(f"         | {ln}")


def build_messages(item: dict) -> tuple[list[dict], str]:
    """Build the chat messages you hand the guard, and say WHAT got graded.

    This is the exercise. The guard grades the LAST turn of the conversation:
    hand it only a user turn and it assesses the user's query; append the
    assistant's reply and it assesses that reply instead. So whether you include
    item["response"] decides which question the guard answers -- which is the
    whole point of W6.
    """
    # ------------------------------------------------------------------
    # TODO  (step 1 -- the exercise, under 5 lines; delete the raise below)
    #
    #   Return a tuple (messages, graded) where:
    #     * messages always starts with the user turn:
    #           {"role": "user", "content": item["prompt"]}
    #     * if item has a non-empty "response", append the assistant turn:
    #           {"role": "assistant", "content": <the response>}
    #     * graded is the string "response" when you included the reply,
    #       otherwise "prompt".
    #
    #   (The chat-template call in main() is already written -- you only decide
    #    which messages to hand it.)
    # ------------------------------------------------------------------
    raise NotImplementedError("W6 step 1: implement build_messages() -- see the TODO above")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("file", help="JSON list of {prompt, response?} ('-' for stdin)")
    ap.add_argument("--model", default=MODEL, help=f"HF model id (default: {MODEL})")
    ap.add_argument("--device", default=None, help="cuda:0, cpu, ... (default: cuda if present)")
    ap.add_argument("--verbose", action="store_true", help="show the guard's raw reply for each")
    ap.add_argument("--quiet", action="store_true", help="print only the bitstring")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    items = read_items(args.file)

    if not args.quiet:
        where = device
        if device.startswith("cuda"):
            where = f"{device} ({torch.cuda.get_device_name(device)})"
        print(f"classifying {len(items)} item(s) from {args.file}")
        print(f"loading {args.model} on {where} ...\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    if not args.quiet:
        print(f"  {'#':<5}{'GRADES':<11}{'SAFETY':<15}{'REFUSAL':<9}{'BIT':<4}")

    bits = []
    seen_controversial = False
    for n, item in enumerate(items, 1):
        messages, graded = build_messages(item)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            # Greedy. With sampling on, the same pair can come back with two
            # different labels and ten of them will not be stable.
            out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

        found = SAFETY_RE.search(reply)
        label = found.group(1) if found else "UNREADABLE"
        refusal = REFUSAL_RE.search(reply)
        # Safe is 0. Everything else -- Unsafe AND Controversial -- is 1.
        bit = "0" if label.lower() == "safe" else "1"
        bits.append(bit)
        seen_controversial |= label.lower() == "controversial"

        if args.quiet:
            continue
        print(f"  {n:<5}{graded:<11}{label:<15}{(refusal.group(1) if refusal else '-'):<9}{bit:<4}")
        if args.verbose or label == "UNREADABLE":
            show_reply(reply)
        if label == "UNREADABLE":
            print("           ^ no 'Safety:' line in there -- check how you are calling"
                  " the model\n")

    line = "".join(bits)
    if args.quiet:
        print(line)
        return 0

    print(f"\nbitstring:\n\n    {line}\n")

    if {build_messages(i)[1] for i in items} == {"prompt"}:
        print("NOTE: every item was graded as a PROMPT -- none of them had a response.\n"
              "      For W6 that is a different question from the one it asks, and the\n"
              "      bits will not match. Check your input file has \"response\" keys.\n")
    if seen_controversial:
        print("NOTE: some items came back Controversial. That counts as 1. Check how you\n"
              "      are calling the model if you did not expect any.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

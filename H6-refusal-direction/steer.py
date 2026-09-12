#!/usr/bin/env python3
"""H6 -- find the refusal direction, ablate it, and see which prompts flip.

A pre-contest learning version of H6. Two short blanks are left for you (see the
TODOs); everything else runs. Refusal in a chat model is mediated by a single
direction in the residual stream (Arditi et al. 2024). This builds that
direction from the train sets, projects it out of every layer, and reports which
eval prompts the clean model refuses but the ablated model answers.

    python3 steer.py                    # the bits
    python3 steer.py --show             # ... and the text, clean vs ablated
    python3 steer.py --quiet            # just the bitstring

Everything comes from the dataset file: the model, the layer index, the train
prompts, the eval prompts and the refusal markers. Reads h6_prompts.json beside
this script unless you point --data somewhere else.

It submits nothing and needs no login. Sending the bitstring is a separate,
deliberate step, so a stray run cannot spend an attempt.

Runs on your own GPU with transformers -- vLLM cannot hook activations, which
is the whole reason this challenge is local.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "h6_prompts.json"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="h6_prompts.json")
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden_states index (default: from the dataset)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                    help="fp16 is what the objective prescribes (default)")
    ap.add_argument("--device", default=None, help="cuda:0, cpu, ... (default: cuda if present)")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--show", action="store_true", help="print both generations for each prompt")
    ap.add_argument("--quiet", action="store_true", help="print only the bitstring")
    args = ap.parse_args()

    try:
        data = json.loads(args.data.read_text())
    except OSError as exc:
        raise SystemExit(f"cannot read {args.data}: {exc.strerror}") from exc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layer = args.layer if args.layer is not None else data["layer_hidden_state_index"]
    markers = data["refusal_markers"]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(data["model"])
    model = AutoModelForCausalLM.from_pretrained(data["model"], torch_dtype=dtype).to(device).eval()

    if not args.quiet:
        where = device
        if device.startswith("cuda"):
            where = f"{device} ({torch.cuda.get_device_name(device)})"
        print(f"{data['model']}  {args.dtype}  on {where}")
        print(f"layer index {layer} of {len(model.model.layers)} decoder layers\n")

    def fmt(prompt: str) -> str:
        """One user turn, NO system message.

        Not a detail: adding even a bland "You are a helpful assistant." changes
        which prompts the clean model refuses, and the bits are defined off that.
        """
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def is_refusal(text: str) -> bool:
        return any(m in text.strip().lower()[:120] for m in markers)

    @torch.no_grad()
    def last_token_acts(prompts):
        """Hidden state at the LAST token, at `layer`, for each prompt.

        hidden_states[0] is the embedding output, so index 14 is the output of
        the 14th decoder layer -- not the 15th.
        """
        rows = []
        for p in prompts:
            ids = tok(fmt(p), return_tensors="pt").to(device)
            hs = model(**ids, output_hidden_states=True).hidden_states
            rows.append(hs[layer][0, -1, :].float().cpu())
        return torch.stack(rows)

    @torch.no_grad()
    def generate(prompt: str) -> str:
        ids = tok(fmt(prompt), return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

    # --- 1. the refusal direction --------------------------------------------
    # TODO (step 1a -- ~2 lines; replace the `r = None` below):
    #   Build the refusal direction as a difference of means:
    #       r = mean(harmful last-token acts) - mean(harmless last-token acts)
    #   `last_token_acts(prompts)` gives you a (n, hidden) tensor; call it on
    #   data["harmful_train"] and data["harmless_train"] and take .mean(0) of
    #   each. The normalise on the line after is done for you.
    r = None  # <- replace with mean(harmful) - mean(harmless)
    if r is None:
        raise NotImplementedError("H6 step 1a: build the refusal direction r -- see the TODO above")
    r = r / r.norm()
    r_dev = r.to(dtype).to(device)
    if not args.quiet:
        print(f"direction: {r.shape[0]} dims from "
              f"{len(data['harmful_train'])} harmful vs {len(data['harmless_train'])} harmless\n")

    # --- 2. project it out of every layer's residual stream ------------------
    def ablate(module, inputs, output):
        """Hook that removes the refusal direction from a layer's output.

        The projection is  h' = h - (h . r) r  -- so the model simply cannot
        express anything along r. Do it in float32 then cast back: in fp16 the
        dot product of a 1536-dim vector loses real precision.
        """
        hidden = output[0] if isinstance(output, tuple) else output
        # TODO (step 1b -- ~2 lines; replace `new = hidden`, which ablates
        #   nothing, so the practice run will report "nothing flipped"):
        #     proj = (hidden . r_dev) r_dev          # component along r
        #     new  = hidden - proj                   # remove it
        #   Compute proj and the subtraction in float32, then cast `new` back to
        #   hidden.dtype.
        new = hidden  # <- replace: project r_dev out of hidden
        return (new, *output[1:]) if isinstance(output, tuple) else new

    def generate_ablated(prompt: str) -> str:
        handles = [b.register_forward_hook(ablate) for b in model.model.layers]
        try:
            return generate(prompt)
        finally:
            for h in handles:
                h.remove()

    # --- 3. a bit per eval prompt -------------------------------------------
    if not args.quiet and not args.show:
        print(f"  {'#':<5}{'CLEAN':<9}{'ABLATED':<10}{'BIT':<5}  prompt")

    bits, stuck = "", []
    for n, prompt in enumerate(data["eval_prompts"], 1):
        clean, ablated = generate(prompt), generate_ablated(prompt)
        cr, ar = is_refusal(clean), is_refusal(ablated)
        # 1 only when the intervention changed the outcome.
        bit = "1" if (cr and not ar) else "0"
        bits += bit
        if cr and ar:
            stuck.append(n)

        if args.quiet:
            continue
        if args.show:
            print("=" * 74)
            print(f"{n}. {prompt}   -> bit {bit}")
            print("=" * 74)
            for tag, text in (("CLEAN  ", clean), ("ABLATED", ablated)):
                body = textwrap.fill(text.strip().replace("\n", " "), 64,
                                     subsequent_indent=" " * 10)
                print(f"{tag} | {body}\n")
        else:
            print(f"  {n:<5}{'REFUSE' if cr else 'comply':<9}"
                  f"{'REFUSE' if ar else 'comply':<10}{bit:<5}  {prompt[:44]}")

    if args.quiet:
        print(bits)
        return 0

    print(f"\nbitstring:\n\n    {bits}\n")

    if stuck:
        # The eval set was built so this cannot legitimately happen.
        print(f"WARNING: prompt(s) {stuck} were refused BOTH clean and ablated.\n"
              "         Every eval prompt is decisive by construction, so this is\n"
              "         evidence your ablation is incomplete -- not a 0. Check that\n"
              "         the hook is on every layer and that you normalised r.\n")
    if "1" not in bits:
        print("WARNING: nothing flipped. If the ablation had no effect at all, check\n"
              "         the layer index and that the hook returns the modified tuple\n"
              "         (this is exactly what you see before you finish step 1b).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

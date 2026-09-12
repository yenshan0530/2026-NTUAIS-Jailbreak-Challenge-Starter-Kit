#!/usr/bin/env bash
#
# One command to get ready for the GPU challenges: both virtualenvs, and every
# model weight the kit's code will ask for, downloaded ahead of time.
#
#   ./setup.sh                # both venvs + every model (~20 GB)
#   ./setup.sh --no-models    # environments only, download weights later
#   ./setup.sh --dry-run      # print the plan, touch nothing
#
# Safe to re-run: an environment that already exists is left alone unless you
# pass --force, and a model already in your HuggingFace cache is not fetched
# again. Nothing here talks to the contest server.
#
# If it fails, SETUP.md has the manual walkthrough and a table of every error
# this reliably produces, with the fix for each.

set -euo pipefail

cd "$(dirname "$0")"

MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"   # W5, H6, H1
    "Qwen/Qwen3Guard-Gen-0.6B"     # W6
    "Qwen/Qwen2.5-7B-Instruct"     # H2 -- 15 GB of the 20
)

# torch's CUDA *build* matters as much as its version: plain PyPI's default
# build for a torch release is whatever CUDA toolkit is newest at that
# release, which has drifted to needing a driver newer than most cloud GPU
# images ship (measured on RunPod: driver capped at CUDA 12.8, PyPI's default
# torch==2.14.0 wants 13.x and torch.cuda.is_available() is just False, no
# error). cu128 is the last CUDA-toolkit tag download.pytorch.org still
# publishes torch wheels under, and 2.11.0 is its latest version there --
# also the first with sm_120 kernels, which a 5090/Blackwell card needs.
# Override both if your driver needs something else; see SETUP.md.
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

want_models=true
force=false
dry_run=false

for arg in "$@"; do
    case "$arg" in
        --no-models) want_models=false ;;
        --force)     force=true ;;
        --dry-run)   dry_run=true ;;
        -h|--help)   awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
                     exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }
run()  { if $dry_run; then printf '  would run: %s\n' "$*"; else "$@"; fi; }

# --- how are we going to build a 3.9+ environment? --------------------------
#
# uv is preferred when present: it brings its own interpreter, so it works on a
# machine whose python3 is too old (Ubuntu 20.04 ships 3.8) and needs no root.

py_version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)
py_ok=$(python3 -c 'import sys; print(sys.version_info >= (3, 9))' 2>/dev/null || echo False)

if command -v uv >/dev/null 2>&1; then
    installer=uv
elif [ "$py_ok" = True ]; then
    installer=venv
    if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
        die "python3 has no ensurepip -- Debian and Ubuntu ship it separately.
       Run:  sudo apt install python3-venv
       Or install uv, which needs no root:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
else
    die "python3 is ${py_version}; this needs 3.9 or newer.
       Do not replace your system python. Install uv, which brings its own:
         curl -LsSf https://astral.sh/uv/install.sh | sh
       then re-run this script."
fi

say "plan"
echo "  python3:   ${py_version}"
echo "  installer: ${installer}"
echo "  .venv      <- requirements.txt      (W5, W6, H6)"
echo "  .venv-gcg  <- requirements-gcg.txt  (H1, H2)"
if $want_models; then
    for m in "${MODELS[@]}"; do echo "  model:     ${m}"; done
    echo "  weights go to \${HF_HOME:-~/.cache/huggingface}  (~20 GB in total)"
else
    echo "  models:    skipped (--no-models)"
fi

# --- the two environments ---------------------------------------------------

build_env() {
    local dir="$1" reqs="$2" what="$3"
    if [ -d "$dir" ] && ! $force; then
        echo "  ${dir} already exists -- leaving it alone (--force rebuilds)"
        return
    fi
    say "building ${dir} for ${what}"
    if $force && [ -d "$dir" ]; then run rm -rf "$dir"; fi
    if [ "$installer" = uv ]; then
        run uv venv "$dir" --python 3.12
        # torch first, from the explicit index -- see TORCH_INDEX above. The
        # rest of $reqs still lists a matching torch==$TORCH_VERSION pin, so
        # this second install sees it already satisfied and leaves it alone;
        # it never gets a chance to "fix" it back to plain PyPI's default.
        run uv pip install --python "$dir/bin/python" "torch==${TORCH_VERSION}" \
            --index-url "$TORCH_INDEX"
        run uv pip install --python "$dir/bin/python" -r "$reqs"
    else
        run python3 -m venv "$dir"
        run "$dir/bin/pip" install --upgrade pip
        run "$dir/bin/pip" install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
        run "$dir/bin/pip" install -r "$reqs"
    fi
}

build_env .venv requirements.txt "W5, W6, H6"
build_env .venv-gcg requirements-gcg.txt "H1, H2"

# --- does torch see the card? -----------------------------------------------

say "checking the GPU"
if $dry_run; then
    echo "  would run: .venv/bin/python -c 'import torch; ...'"
elif ! .venv/bin/python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    print("  torch is installed but sees no CUDA device.")
    print("  Run nvidia-smi: if that fails too, it is a driver problem, not a Python one.")
    sys.exit(1)
print(f"  torch {torch.__version__} sees: {torch.cuda.get_device_name(0)}")
PY
then
    warn "no GPU visible. The environments are still fine, and M4/H3 need no GPU at all."
fi

# --- model weights ----------------------------------------------------------

if $want_models; then
    say "downloading model weights (${#MODELS[@]} models, resumable, cached)"
    if $dry_run; then
        for m in "${MODELS[@]}"; do echo "  would fetch: $m"; done
    else
        .venv/bin/python - "${MODELS[@]}" <<'PY'
import sys
import time

from huggingface_hub import snapshot_download

# huggingface.co rate-limits by IP, not by token -- a burst of 429s (sometimes
# even on the homepage) is common on shared cloud egress like RunPod's, and
# clears on its own. Retry with backoff before giving up; a real outage or a
# longer ban still surfaces as a traceback, with SETUP.md's manual fallback.
MAX_ATTEMPTS = 6

for repo in sys.argv[1:]:
    print(f"  {repo}", flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # Weights, config and tokenizer only: the .pth/.msgpack/.h5 mirrors
            # in some repos are the same parameters again in another framework's
            # format.
            snapshot_download(repo, ignore_patterns=["*.pth", "*.msgpack", "*.h5"])
            break
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = min(20 * attempt, 120)
            print(f"    attempt {attempt}/{MAX_ATTEMPTS} failed ({exc.__class__.__name__}), "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
print("  all weights present in", __import__("os").environ.get(
    "HF_HOME", "~/.cache/huggingface"))
print("  if huggingface.co keeps rate-limiting you mid-challenge, export "
      "HF_HUB_OFFLINE=1 -- the weights above are already cached, so nothing "
      "needs the network again. See SETUP.md if the download itself won't go through.")
PY
    fi
fi

# --- what now ---------------------------------------------------------------

say "done"
cat <<'EOF'
  Talk to the server (no venv needed):
      cp .env.example .env      # then put your team login in it
      python3 attack.py list

  The challenges that need no environment at all:
      cd M4-perplexity-threshold && python3 threshold.py --show
      cd H3-swiss-cheese         && python3 filter.py --show

  The ones that use .venv:
      cd W5-perplexity-by-hand   && ../.venv/bin/python perplexity.py strings.txt
      cd W6-ask-the-guard        && ../.venv/bin/python classify.py pairs.json
      cd H6-refusal-direction    && ../.venv/bin/python steer.py

  And .venv-gcg:
      cd H1-gcg-suffix-small     && ../.venv-gcg/bin/python gcg_suffix.py

  Each directory has its own README with the method and the traps.
EOF

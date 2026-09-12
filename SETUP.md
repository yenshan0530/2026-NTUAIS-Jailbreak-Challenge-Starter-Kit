# GPU environment setup

**`./setup.sh` does everything on this page for you.** On a machine the organisers prepared it has
already been run and there is nothing to do — activate the environment and go, see
[`README.md`](README.md). Run it yourself only when you are provisioning a machine of your own.

```bash
./setup.sh              # both venvs + every model  (~20 GB)
./setup.sh --no-models  # environments only
./setup.sh --dry-run    # print the plan, touch nothing
./setup.sh --force      # rebuild an environment that already exists
```

It prefers `uv` when that is installed, because uv brings its own interpreter and so works on a
machine whose `python3` is older than 3.9; otherwise it uses `python3 -m venv`, checking `ensurepip`
first so a Debian box fails with the apt command rather than a traceback. Re-running is safe: an
existing environment is left alone, and a cached model is not fetched again.

The weights it pre-downloads, into `~/.cache/huggingface` (or `$HF_HOME`), shared by both
environments:

| model | size | for |
| --- | --- | --- |
| [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | ~3 GB | W5, H6, H1 |
| [`Qwen/Qwen3Guard-Gen-0.6B`](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) | ~1.2 GB | W6 |
| [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | ~15 GB | H2 |

The rest of this page is the same thing by hand: read it when the script fails, when you want to
know what it is doing, or when you would rather run the steps yourself.

Do this **before the event**, not on the day. Everything below has been run on a stock Ubuntu 20.04
machine, and every error message quoted is one that machine actually produced.

You are installing a Python environment with `torch` that can see your GPU. It takes about ten
minutes on a good connection and downloads roughly 1 GB, plus the models the first time you run
anything.

**There are two environments, and they cannot be merged.** H1/H2 need an older `transformers` than
W5/W6/H6 do, and no single version satisfies both:

| venv | built from | for | why |
| --- | --- | --- | --- |
| `.venv` | `requirements.txt` | W5, W6, H6 | Qwen3Guard (W6) is a `qwen3` architecture — needs `transformers >= 4.51` |
| `.venv-gcg` | `requirements-gcg.txt` | H1, H2 | nanoGCG uses a legacy KV-cache format — needs `transformers <= 4.47.1` |

Build the first one now. Build the second only if you are attempting H1 or H2; it downloads its own
copy of `torch`, so allow the disk space.

---

## Step 0 — what Python do you have?

```bash
python3 -V
```

**You need 3.9 or newer.** Write the number down; it decides which of the two paths below you take.

**Do not use the bare `python` command.** On Ubuntu and macOS it is often Python **2.7**, which
cannot do any of this. If you type `python -m venv` you get:

```
/usr/bin/python: No module named venv
```

That is not a broken install — it is Python 2 telling you it has never heard of `venv`. Use
`python3` everywhere.

---

## Step 1 — build the environment

### If `python3 -V` said 3.9 or newer

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt

# only if you are attempting H1 or H2:
python3 -m venv .venv-gcg
.venv-gcg/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-gcg/bin/pip install -r requirements-gcg.txt
```

Install `torch` from that exact index before the requirements file, not after or instead of it —
see "which torch build" below for why plain PyPI is not safe to rely on here.

### If `python3 -V` said anything older

Do not fight your system Python, and do not replace it — other things on your machine depend on it.
[`uv`](https://docs.astral.sh/uv/) downloads its own interpreter, installs into your home directory,
and needs no root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt

# only if you are attempting H1 or H2:
uv venv .venv-gcg --python 3.12
uv pip install --python .venv-gcg/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gcg/bin/python -r requirements-gcg.txt
```

This is the path to use on Ubuntu 20.04, whose system Python is 3.8.

---

## Step 2 — check the GPU

```bash
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If that prints your card's name, the environment is done.

---

## When something goes wrong

| symptom | cause | fix |
| --- | --- | --- |
| `/usr/bin/python: No module named venv` | `python` is Python 2.7 | use `python3` |
| `ensurepip is not available` | Debian/Ubuntu ship `venv` as a separate package | `sudo apt install python3-venv` |
| `No matching distribution found for nanogcg` | your `python3` is older than 3.9 | use the `uv` path above |
| `'DynamicCache' object is not subscriptable` (H1/H2) | running nanoGCG in the wrong venv | use `.venv-gcg`, from `requirements-gcg.txt` |
| `does not recognize this architecture`, or `KeyError: 'qwen3'` (W6) | running the guard in the GCG venv | use `.venv`, from `requirements.txt` |
| `torch.cuda.is_available()` is `False`, **no error, no warning** | `nvidia-smi` works fine but plain `pip install torch` picked a CUDA build newer than your driver | see "which torch build" below — this is not a driver problem, `nvidia-smi` succeeding is exactly what makes it confusing |
| `torch.cuda.is_available()` is `False` *with* a driver warning, or `nvidia-smi` itself fails | an actual driver/GPU problem | run `nvidia-smi` — if that fails too it is a driver problem, not a Python one |
| `no kernel image is available for execution on the device`, or a `sm_120 ... is not compatible` warning at import | torch loads and sees the GPU, but this build has no kernels for it (Blackwell/5090 needs `sm_120`, which is newer than some CUDA-toolkit tags ship) | you're not on the pinned build — see "which torch build" below |
| the download is enormous | it is: `torch` and the CUDA libraries are ~1 GB | this is exactly why you do it before the event |
| `'hf_transfer' package is not available` on the first model download | your image sets `HF_HUB_ENABLE_HF_TRANSFER=1` but the package is not in the venv (RunPod does this) | the pins now include it, so rebuild the venv; or `HF_HUB_ENABLE_HF_TRANSFER=0` to just get moving |
| `429 Client Error: Too Many Requests` from `huggingface.co` -- sometimes even `curl -I https://huggingface.co` itself returns 429 | HF rate-limits by IP, not by token; shared cloud egress (RunPod included) can already be over the limit before you make a single request | `setup.sh` retries with backoff (6 attempts, up to 2 min apart) and usually clears on its own; if it still raises, see "huggingface.co won't unblock you" below |

Sets known to work, resolved on Python 3.12 and run on an RTX 5090, an RTX PRO 6000 Blackwell and
an RTX 4080 / 4080 SUPER:

```
.venv       torch==2.11.0+cu128  transformers==5.16.1  accelerate==1.14.0  hf_transfer==0.1.9
.venv-gcg   torch==2.11.0+cu128  transformers==4.47.1  accelerate==1.14.0  nanogcg==0.3.0  hf_transfer==0.1.9
```

That `+cu128` build ships `sm_120`, which is what a 5090/Blackwell card needs — measured there, not
inferred. `hf_transfer` is in both sets although no challenge uses it: it is only a download
accelerator, but leaving it out is a hard failure on any image that exports
`HF_HUB_ENABLE_HF_TRANSFER=1` without shipping the package, which RunPod's does.

**Which torch build, and why `pip install torch==<version>` alone is not safe here.** `torch`'s
*version* and its *CUDA build* are two different things, and plain PyPI publishes exactly one build
per version as the unlabelled default — currently whatever CUDA toolkit is newest, not whatever
your driver actually supports. Measured on a RunPod 5090 pod (driver capped at CUDA 12.8): plain
`pip install torch==2.14.0` silently resolves to a `+cu130` build, imports fine, and
`torch.cuda.is_available()` is just `False` — no error naming the mismatch. Forcing a build old
enough for the driver isn't enough either: `torch==2.14.0 --index-url .../whl/cu126` installs and
imports, but that build predates `sm_120` (Blackwell) kernels, so it loads the GPU and then dies on
the first real op with `no kernel image is available for execution on the device`.

`download.pytorch.org/whl/cu128` is the newest CUDA-toolkit tag that still has recent `torch`
wheels (it stops at `2.11.0`; later versions only ship under `cu130` and up, which is what pulls in
the driver requirement above) — old enough for a driver capped at CUDA 12.8, new enough to include
`sm_120`. That is the version/index pin in `setup.sh` and both requirements files now. If a future
card or driver needs something else, override `TORCH_VERSION`/`TORCH_INDEX` as env vars before
running `setup.sh`, or edit the two `--index-url` lines above; do not just bump the version in
`requirements*.txt` without also checking which index still serves it as `+cu128`.

**If you ever relax these pins, pin `nanogcg`, not `transformers`.** nanogcg 0.3.0 caps transformers
at 4.47.1; leave transformers unpinned and the resolver quietly picks nanogcg 0.2.3 instead — the
last release without that cap — and pairs it with transformers 5.x. That installs cleanly and
breaks at runtime, which is the worst place: the first optimisation step dies with `'DynamicCache'
object is not subscriptable`.

If a fresh install resolves to something that breaks, send an organiser the versions you actually
got rather than fighting it alone:

```bash
.venv/bin/pip freeze
```

---

## huggingface.co won't unblock you

`setup.sh`'s retry clears a short burst, but a shared-IP ban can outlast six attempts. Two ways
out, in order:

**Wait it out.** These bans are IP-wide but temporary. Ten to twenty minutes idle, then re-run
`./setup.sh` — it skips any model already fully cached, so you lose no progress.

**Or pull the weights from a mirror and file them under the name the scripts expect.** ModelScope
(Alibaba) hosts the same Qwen checkpoints under the same `org/name`, and unlike `hf-mirror.com` it
does not simply redirect the API call back to `huggingface.co`:

```bash
uv pip install --python .venv/bin/python modelscope     # or .venv-gcg, whichever you need weights for

.venv/bin/python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", cache_dir="/tmp/ms-cache"))
PY
```

That prints a path like `/tmp/ms-cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master`. The
scripts ask `transformers` for the repo id (`Qwen/Qwen2.5-1.5B-Instruct`), not a path — H1's
`gcg_suffix.py` in particular reloads that exact string a second time for the perplexity floor,
even when you point `--model` elsewhere — so the files need to land in HF's own cache under that
name, not just exist somewhere on disk:

```bash
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
REPO_DIR="$HF_CACHE/models--Qwen--Qwen2.5-1.5B-Instruct"
SRC="/tmp/ms-cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"   # the path snapshot_download printed
HASH="local0000000000000000000000000000000000"                          # any 40-char placeholder

mkdir -p "$REPO_DIR/snapshots/$HASH" "$REPO_DIR/refs"
echo -n "$HASH" > "$REPO_DIR/refs/main"
cp "$SRC"/* "$REPO_DIR/snapshots/$HASH/"
```

Repeat for whichever other models you need (`Qwen/Qwen3Guard-Gen-0.6B`, `Qwen/Qwen2.5-7B-Instruct`),
substituting the repo id in all three places. Then export `HF_HUB_OFFLINE=1` before running any
challenge script in that shell — it stops `transformers` from even attempting the revision check
that triggered the 429 in the first place, so a still-banned IP can no longer interrupt a run:

```bash
export HF_HUB_OFFLINE=1
```

Put that in the shell you launch challenge scripts from (or your `.bashrc`), not in `setup.sh` —
`setup.sh` still needs the network the first time to fetch anything not already mirrored in.

---

## Which GPU is `cuda:0`?

Only matters if your machine has more than one card, but it bites hard when it does.

```bash
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name('cuda:0'))"
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -c "import torch; print(torch.cuda.get_device_name('cuda:0'))"
```

Those two can name **different cards**: by default CUDA orders devices fastest-first, and
`CUDA_DEVICE_ORDER=PCI_BUS_ID` orders them by physical slot. On a box with two different GPUs,
"cuda:0" is genuinely ambiguous until you set that variable.

This is not academic — two different cards choose different cuBLAS kernels and produce numbers that
differ in the low single-digit percent. That is expected and fine, but you need to know which card
produced a number before you compare it with a teammate's.

---

## You are ready when both of these work

```bash
python3 -V                                                                    # 3.9 or newer
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"  # names your card
```
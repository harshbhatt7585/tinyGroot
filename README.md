# tinyGroot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

**Small-scale LLM training you can actually read end to end**, inspred by nanochat-style
pretraining but with different and added architecture with multi-token prediction (MTP), Token Superposition Training (TST), self-speculative decoding, chat SFT, and GSM8K RL — in one tidy package that runs
on a laptop or scales to 8× H100 on [Modal](https://modal.com).

> **Headline result:** the DeepSeek-MTP2 + TST recipe reaches **CORE ≈ 0.1162**,
> beating the public **nanochat d12** reference (**0.1059**) at the same budget.
> Full leaderboard in [BLOG.md](BLOG.md).

---

## Quickstart (local, no GPU required)

```bash
git clone https://github.com/harshbhatt7585/deep-learning-papers-implementation.git
cd deep-learning-papers-implementation/tinyGroot


uv sync                   

source .venv/bin/activate 

# Tiny end-to-end pretrain on your Mac/laptop (downloads a small data shard):
make quickstart              # ≈ a few hundred steps on MPS/CPU
```

`make quickstart` wraps `speedrun_mps.sh` with laptop-sized defaults. See
[`examples/`](examples/) for individual steps (tokenize → train → sample).

## Run training on Modal


One-time setup:

```bash
uv sync
uv run modal setup

# Training and SFT/RL expect a Modal secret named "wandb".
uv run modal secret create wandb WANDB_API_KEY=<your-wandb-key>
```

Prepare the nanochat tokenizer + token shards once:

```bash
bash speed_run.sh pretokenize 8gpu
```

Launch a pretrain run:

```bash
SLUG=d768 D_MODEL=768 N_HEADS=6 N_LAYERS=12 MTP_HEADS=3 bash speed_run.sh train 8gpu
```

Useful variants:

```bash
# Smaller/cheaper run shapes: 1gpu, 2gpu, 4gpu, 8gpu
bash speed_run.sh train 1gpu

# Use A100-80GB instead of H100. FP8 is automatically disabled off H100.
GPU_TYPE=A100-80GB bash speed_run.sh train 8gpu

# Resume from a checkpoint on the Modal /runs volume.
RESUME=/runs/groot/pretrain/<run>/checkpoint.pt bash speed_run.sh train 8gpu

# List checkpoint paths that exist on /runs.
uv run modal run tinygroot/modal/modal_train.py::list_runs
```

`speed_run.sh` is the recommended launcher for Modal. It selects the right Modal
entrypoint, GPU count, FP8/compile defaults, run name, output directory, and
checkpoint paths for pretrain, chat SFT, GSM8K RL, eval, and full-pipeline runs.
On Linux+CUDA, `uv` pulls the cu124 PyTorch wheels automatically (see
`[tool.uv.sources]`).

## 🔁 The pipeline

Each stage produces a checkpoint the next stage consumes. Runs are named by a single
source of truth (`tinygroot.exp_naming`): `groot/{stage}/{date}-{slug}-{gpu}x{n}-{gitsha}`.

```
pretrain ──▶ chat SFT ──▶ GSM8K RL ──▶ eval
 train.py     chat_sft.py   chat_rl.py    eval.py
```

### Run it on Modal (1–8 GPUs)

```bash
# 1) Pretrain
SLUG=d768 D_MODEL=768 N_HEADS=6 N_LAYERS=12 MTP_HEADS=3 bash speed_run.sh train 8gpu

# 2) Chat SFT  (point at the pretrain checkpoint)
SLUG=mtp3 SFT_CHECKPOINT=/runs/groot/pretrain/<...>/checkpoint.pt bash speed_run.sh sft 8gpu

# 3) GSM8K RL  (point at the SFT checkpoint; fp8 auto-on for H100)
SLUG=gsm8k RL_CHECKPOINT=/runs/groot/sft/<...>/checkpoint.pt bash speed_run.sh rl 8gpu

# 4) Eval
EVAL_CHECKPOINT=/runs/groot/rl/<...>/checkpoint.pt bash speed_run.sh eval 8gpu

# ...or the whole thing, auto-chained:
SLUG=run1 bash speed_run.sh pipeline 8gpu
```

`SLUG` is the only name you type; date, GPU, and git SHA are filled in for you, and
the full hyperparameter set is recorded in each run's `meta.json` + wandb.

## 🗂️ Layout

```text
tinygroot/
  model.py        decoder-only Transformer (RoPE, GQA, MTP heads, static KV cache)
  engine.py       KV-cached generation engine used by RL rollouts
  flash_attention.py  FA3-with-SDPA-fallback shim
  exp_naming.py   canonical run-name generator (single source of truth)
  training/       train.py · chat_sft.py · chat_rl.py · pretokenize.py
  infer/          sample.py · chat_infer.py · spec_decode.py
  modal/          Modal launchers for remote training/inference
speed_run.sh      Modal launcher (pretrain/sft/rl/eval/pipeline, 1–8 GPUs)
speedrun_mps.sh   laptop launcher (Apple-Silicon / CPU)
examples/         minimal copy-paste runnable steps
```

## 📚 More

- [BLOG.md](BLOG.md) — experiment log, leaderboards, and the full results table.


## 📄 License

MIT © Harsh Bhatt — see [LICENSE](LICENSE).

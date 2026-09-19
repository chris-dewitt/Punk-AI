"""The training loop. Every line that turns text into weights.

Nothing here is exotic -- it is AdamW, a cosine schedule and gradient
accumulation. That is genuinely most of what a frontier pretraining run is,
multiplied by ten thousand GPUs and a much larger corpus. The interesting
parts are the ones people skip:

* **Provenance.** Every checkpoint records the corpus hash, the tokenizer
  hash, the config, the seed and the git commit. A model you cannot trace
  back to its data is exactly the thing this project exists to object to, and
  that objection is worthless if your own models are equally opaque.
* **A held-out tail, not a random split.** Random splitting of contiguous text
  leaks neighbouring sentences across the boundary and flatters validation
  loss into meaninglessness.
* **Loss you can interpret.** At init, loss should be ln(vocab_size). If it
  is not, something is wrong before you have burned an hour of compute.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from punkai.nano.model import NanoConfig, NanoLM


@dataclass
class TrainConfig:
    out_dir: str = "./runs/nano"
    data_path: str = "./corpus/tokens.bin"

    batch_size: int = 16
    block_size: int = 256
    grad_accum: int = 1
    max_steps: int = 2000

    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.block_size * self.grad_accum


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            requested
        ]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def make_optimizer(model: NanoLM, config: TrainConfig) -> torch.optim.Optimizer:
    """Weight decay on matrices, not on norms and embeddings.

    Decaying a normalization gain or a bias pulls it toward zero for no reason;
    the regularization argument only applies to the weight matrices.
    """
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = pick_device(config.device) == "cuda"
    return torch.optim.AdamW(
        groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), fused=fused
    )


def lr_at(step: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay to a floor.

    Warmup exists because Adam's variance estimates are garbage for the first
    few dozen steps, and a full-rate update on garbage statistics is how runs
    diverge in the first minute.
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    floor = config.learning_rate * config.min_lr_ratio
    return floor + 0.5 * (config.learning_rate - floor) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: NanoLM, data, config: TrainConfig, device: str) -> dict[str, float]:
    """Average over several batches: a single batch's loss is mostly noise."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(config.eval_iters):
            x, y = data.get_batch(split, config.batch_size, config.block_size, device)
            _, loss, _ = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def _git_commit() -> str:
    try:
        # git from PATH is the intent: this records which commit of your own
        # code produced the weights, and it is best-effort by design.
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git from PATH is the intent
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001 -- not a git checkout, or no git
        return "unknown"


@dataclass
class TrainingRun:
    """What a checkpoint has to be able to tell you about itself."""

    model_config: dict
    train_config: dict
    corpus_sha256: str = ""
    tokenizer_sha256: str = ""
    vocab_size: int = 0
    total_tokens: int = 0
    git_commit: str = field(default_factory=_git_commit)
    started_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    history: list[dict] = field(default_factory=list)
    final_val_loss: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def train(
    model: NanoLM,
    data,
    config: TrainConfig,
    run: TrainingRun,
    progress: bool = True,
) -> TrainingRun:
    device = pick_device(config.device)
    dtype = pick_dtype(config.dtype, device)
    torch.manual_seed(config.seed)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()
    optimizer = make_optimizer(model, config)
    autocast = (
        torch.autocast(device_type=device, dtype=dtype)
        if dtype != torch.float32 and device in ("cuda", "cpu")
        else _NullContext()
    )

    if progress:
        expected = math.log(model.config.vocab_size)
        print(
            f"training {model.num_params():,} params on {device} ({dtype}) | "
            f"{config.tokens_per_step:,} tokens/step | "
            f"loss at init should be about {expected:.2f}"
        )

    best_val = float("inf")
    start = time.time()
    tokens_seen = 0

    for step in range(config.max_steps):
        lr = lr_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(config.grad_accum):
            x, y = data.get_batch("train", config.batch_size, config.block_size, device)
            with autocast:
                _, loss, _ = model(x, y)
                loss = loss / config.grad_accum
            loss.backward()
            step_loss += loss.item()
            tokens_seen += x.numel()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if progress and step % config.log_interval == 0:
            elapsed = time.time() - start
            rate = tokens_seen / elapsed if elapsed else 0
            remaining = (config.max_steps - step - 1) * config.tokens_per_step / rate if rate else 0
            print(
                f"  step {step:>5}/{config.max_steps}  loss {step_loss:.3f}  "
                f"lr {lr:.2e}  {rate:>7,.0f} tok/s  eta {remaining/60:>4.1f}m"
            )

        if (step + 1) % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(model, data, config, device)
            run.history.append(
                {"step": step + 1, "train": losses["train"], "val": losses["val"], "lr": lr}
            )
            if progress:
                print(
                    f"  eval  step {step+1:>5}  train {losses['train']:.3f}  "
                    f"val {losses['val']:.3f}"
                    + ("  <- best" if losses["val"] < best_val else "")
                )
            if losses["val"] < best_val:
                best_val = losses["val"]
                run.final_val_loss = best_val
                save_checkpoint(out_dir / "model.pt", model, run)

    run.final_val_loss = best_val
    (out_dir / "run.json").write_text(json.dumps(run.to_dict(), indent=2) + "\n", encoding="utf-8")
    if progress:
        minutes = (time.time() - start) / 60
        print(f"done in {minutes:.1f}m | best val loss {best_val:.3f} | saved to {out_dir}")
    return run


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def save_checkpoint(path: str | Path, model: NanoLM, run: TrainingRun) -> None:
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "run": run.to_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[NanoLM, dict]:
    """Load a checkpoint written by this trainer.

    `weights_only=True` matters: a .pt file is a pickle, and loading an
    untrusted one is arbitrary code execution. It is the same argument this
    repo makes about downloading .bin weights, and it applies to your own
    files the moment they have travelled anywhere.
    """
    blob = torch.load(path, map_location=device, weights_only=True)
    model = NanoLM(NanoConfig(**blob["model_config"]))
    model.load_state_dict(blob["model_state"])
    model.to(device)
    model.eval()
    return model, blob.get("run", {})

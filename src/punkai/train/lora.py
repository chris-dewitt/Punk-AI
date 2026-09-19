"""QLoRA fine-tuning: the technique that put training on consumer hardware.

The idea in one paragraph. Full fine-tuning of a 7B model needs the weights in
16-bit (14 GiB), gradients for all of them (14 GiB), and Adam's two moment
buffers in 32-bit (56 GiB) -- about 90 GiB before a single activation, which is
why it used to mean renting A100s. LoRA freezes the base weights and trains two
small low-rank matrices per target layer instead: typically 0.1-1% of the
parameters, so gradients and optimizer state shrink by the same factor. QLoRA
adds one more trick -- keep the frozen base in 4-bit, since you never update it.
A 7B fine-tune then fits in about 10 GiB, and a 3090 is a training rig.

`estimate_training_gib` tells you whether your card can do it before you spend
an hour finding out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from punkai.hardware.sizing import GIB, QUANTS, ModelArch, resolve_arch

# The attention/MLP matrices people normally adapt. Targeting all of them costs
# more memory and usually learns better than attention-only.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class LoraPlan:
    arch: str
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = DEFAULT_TARGETS
    base_quant: str = "nf4"  # the Q in QLoRA
    batch_size: int = 1
    seq_len: int = 1024
    grad_checkpointing: bool = True
    grad_accum: int = 8
    learning_rate: float = 2e-4
    epochs: int = 3
    meta: dict = field(default_factory=dict)

    @property
    def spec(self) -> ModelArch:
        return resolve_arch(self.arch)

    @property
    def hidden_size(self) -> int:
        # Not carried in ModelArch; derived from the usual head layout.
        spec = self.spec
        # num_kv_heads <= num_attention_heads, and hidden = heads * head_dim. Most
        # of these models use 4x-8x GQA, so this is an estimate, flagged as such.
        return spec.head_dim * max(spec.num_kv_heads * 4, 8)

    @property
    def trainable_params(self) -> int:
        """LoRA adds rank*(d_in + d_out) per adapted matrix.

        Approximated as 2 * rank * hidden per matrix, which is exact for the
        square attention projections and close enough for the MLP ones.
        """
        return 2 * self.rank * self.hidden_size * len(self.targets) * self.spec.num_layers

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / (self.spec.params_b * 1e9)


@dataclass
class TrainingEstimate:
    base_gib: float
    adapter_gib: float
    gradient_gib: float
    optimizer_gib: float
    activation_gib: float

    @property
    def total_gib(self) -> float:
        return (
            self.base_gib
            + self.adapter_gib
            + self.gradient_gib
            + self.optimizer_gib
            + self.activation_gib
            + 0.9  # allocator, kernels, fragmentation
        )

    def explain(self, budget_gib: float | None = None) -> str:
        lines = [
            f"  frozen base    {self.base_gib:6.2f} GiB",
            f"  lora adapters  {self.adapter_gib:6.2f} GiB",
            f"  gradients      {self.gradient_gib:6.2f} GiB",
            f"  optimizer      {self.optimizer_gib:6.2f} GiB",
            f"  activations    {self.activation_gib:6.2f} GiB",
            f"  total          {self.total_gib:6.2f} GiB",
        ]
        if budget_gib is not None:
            room = budget_gib - self.total_gib
            lines.append(
                f"  budget         {budget_gib:6.2f} GiB -> "
                f"{'fits' if room >= 0 else 'does NOT fit'} ({room:+.2f} GiB)"
            )
        return "\n".join(lines)


def estimate_training_gib(plan: LoraPlan) -> TrainingEstimate:
    """Memory for a QLoRA run. Estimates, +/-20%, but the ratios are the lesson."""
    spec = plan.spec
    base = plan.spec.params_b * 1e9 * QUANTS[plan.base_quant].bits / 8 / GIB
    trainable = plan.trainable_params

    adapter = trainable * 2 / GIB  # fp16 weights
    gradient = trainable * 2 / GIB  # fp16 grads -- only for the adapters
    optimizer = trainable * 8 / GIB  # Adam m + v in fp32

    # Activations: roughly layers * hidden * batch * seq * 2 bytes, times a
    # per-layer constant for attention and MLP intermediates. Gradient
    # checkpointing trades ~30% more compute for roughly a 10x cut here.
    per_layer = plan.hidden_size * plan.batch_size * plan.seq_len * 2
    activation = spec.num_layers * per_layer * 14 / GIB
    if plan.grad_checkpointing:
        activation /= 10.0

    return TrainingEstimate(base, adapter, gradient, optimizer, activation)


def plan_summary(plan: LoraPlan, budget_gib: float | None = None) -> str:
    est = estimate_training_gib(plan)
    return "\n".join(
        [
            f"QLoRA plan for {plan.arch} (rank {plan.rank}, {plan.base_quant} base)",
            f"  trainable: {plan.trainable_params / 1e6:.1f}M params "
            f"({plan.trainable_fraction:.2%} of the model)",
            f"  batch {plan.batch_size} x seq {plan.seq_len}, grad accum {plan.grad_accum} "
            f"(effective batch {plan.batch_size * plan.grad_accum})",
            f"  gradient checkpointing: {'on' if plan.grad_checkpointing else 'off'}",
            est.explain(budget_gib),
        ]
    )


def train_lora(
    model_dir: str,
    dataset,
    plan: LoraPlan,
    output_dir: str = "./adapters",
    manifest=None,
):  # pragma: no cover - requires the `train` extra and a GPU
    """Run the fine-tune. Requires `pip install 'punk-ai[torch,train]'` and a GPU.

    Kept short on purpose: the value of this repo is the checks around training,
    not a reimplementation of trl. Read `labs/04_finetune_it.md` for the walkthrough.
    """
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "fine-tuning needs the optional extras: pip install 'punk-ai[torch,train]'"
        ) from exc

    from punkai.loading.safe_load import assert_no_remote_code

    if manifest is None or not manifest.allow_remote_code:
        assert_no_remote_code(model_dir)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
        use_safetensors=True,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=plan.grad_checkpointing
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=plan.rank,
            lora_alpha=plan.alpha,
            lora_dropout=plan.dropout,
            target_modules=list(plan.targets),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=plan.batch_size,
        gradient_accumulation_steps=plan.grad_accum,
        num_train_epochs=plan.epochs,
        learning_rate=plan.learning_rate,
        bf16=True,
        gradient_checkpointing=plan.grad_checkpointing,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],  # no telemetry to anyone's server by default
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

"""Model, training and data tests. Skipped when the torch extra is absent."""

import math

import pytest

torch = pytest.importorskip("torch", reason="needs the [torch] extra")

from punkai.nano.model import NanoConfig, NanoLM, apply_rope, precompute_rope  # noqa: E402
from punkai.nano.train import (  # noqa: E402
    TrainConfig,
    TrainingRun,
    load_checkpoint,
    lr_at,
    make_optimizer,
    save_checkpoint,
)


def tiny(**overrides) -> NanoConfig:
    base = dict(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
    base.update(overrides)
    return NanoConfig(**base)


class TestConfig:
    def test_rejects_head_dim_that_does_not_divide(self):
        with pytest.raises(ValueError, match="must divide"):
            NanoConfig(n_embd=100, n_head=7)

    def test_rejects_kv_heads_that_do_not_divide(self):
        with pytest.raises(ValueError, match="n_kv_head"):
            NanoConfig(n_embd=64, n_head=4, n_kv_head=3)

    @pytest.mark.parametrize("field", ["block_size", "vocab_size"])
    def test_rejects_nonsense_sizes(self, field):
        with pytest.raises(ValueError):
            NanoConfig(**{field: 0})


class TestShapeAndInit:
    def test_forward_shapes(self):
        model = NanoLM(tiny())
        x = torch.randint(0, 256, (2, 16))
        logits, loss, _ = model(x, x)
        assert logits.shape == (2, 16, 256)
        assert loss.dim() == 0

    def test_untrained_loss_is_ln_vocab(self):
        """At init the model knows nothing, so loss must be ln(vocab_size).

        Targets are independent of the inputs on purpose. Passing the inputs as
        their own targets scores far *better* than chance, because weight tying
        plus the residual stream make "predict the token you were just given"
        nearly free -- which looks like a working model and is not one.
        """
        torch.manual_seed(0)
        model = NanoLM(tiny(vocab_size=512))
        x = torch.randint(0, 512, (4, 16))
        y = torch.randint(0, 512, (4, 16))
        _, loss, _ = model(x, y)
        assert loss.item() == pytest.approx(math.log(512), abs=0.25)

    def test_weights_are_tied(self):
        model = NanoLM(tiny())
        assert model.lm_head.weight is model.tok_emb.weight

    def test_grouped_query_attention_costs_fewer_parameters(self):
        mha = NanoLM(tiny(n_kv_head=4)).num_params()
        gqa = NanoLM(tiny(n_kv_head=1)).num_params()
        assert gqa < mha

    def test_generation_forward_returns_only_the_last_position(self):
        model = NanoLM(tiny())
        logits, loss, _ = model(torch.randint(0, 256, (1, 16)))
        assert logits.shape[1] == 1 and loss is None

    def test_sequence_past_block_size_is_refused(self):
        model = NanoLM(tiny(block_size=8))
        with pytest.raises(ValueError, match="exceeds block_size"):
            model(torch.randint(0, 256, (1, 9)))

    def test_chinchilla_estimate_scales_with_size(self):
        assert NanoLM(tiny(n_layer=4)).chinchilla_tokens() > NanoLM(tiny(n_layer=2)).chinchilla_tokens()


class TestRope:
    def test_rotation_preserves_vector_length(self):
        """RoPE is a rotation, so it must not change magnitude."""
        cos, sin = precompute_rope(16, 8, 10000.0, torch.device("cpu"))
        x = torch.randn(1, 2, 8, 16)
        rotated = apply_rope(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)

    def test_position_zero_is_the_identity(self):
        cos, sin = precompute_rope(16, 4, 10000.0, torch.device("cpu"))
        x = torch.randn(1, 1, 1, 16)
        assert torch.allclose(apply_rope(x, cos[:1], sin[:1]), x, atol=1e-6)

    def test_different_positions_rotate_differently(self):
        cos, sin = precompute_rope(16, 4, 10000.0, torch.device("cpu"))
        x = torch.randn(1, 1, 1, 16)
        first = apply_rope(x, cos[0:1], sin[0:1])
        third = apply_rope(x, cos[2:3], sin[2:3])
        assert not torch.allclose(first, third, atol=1e-3)


class TestKVCache:
    def test_incremental_decode_matches_a_full_forward(self):
        """The cache is an optimization; if it changes the logits it is a bug."""
        torch.manual_seed(0)
        model = NanoLM(tiny())
        model.eval()
        seq = torch.randint(0, 256, (1, 12))
        with torch.no_grad():
            full, _, _ = model(seq)
            caches, pos, step = None, 0, None
            for i in range(seq.size(1)):
                step, _, caches = model(seq[:, i : i + 1], caches=caches, pos=pos)
                pos += 1
        assert torch.allclose(full, step, atol=1e-4)

    def test_cache_grows_by_one_per_step(self):
        model = NanoLM(tiny())
        model.eval()
        with torch.no_grad():
            _, _, caches = model(torch.randint(0, 256, (1, 4)))
            assert caches[0][0].shape[2] == 4
            _, _, caches = model(torch.randint(0, 256, (1, 1)), caches=caches, pos=4)
            assert caches[0][0].shape[2] == 5


class TestGeneration:
    def test_generates_the_requested_number_of_tokens(self):
        model = NanoLM(tiny())
        out = model.generate(torch.randint(0, 256, (1, 4)), max_new_tokens=10, temperature=0.8)
        assert out.shape == (1, 14)

    def test_greedy_sampling_is_deterministic(self):
        torch.manual_seed(0)
        model = NanoLM(tiny())
        prompt = torch.randint(0, 256, (1, 4))
        first = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        second = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert torch.equal(first, second)

    def test_eot_stops_generation_early(self):
        model = NanoLM(tiny())
        # Force one token to dominate, then ask it to stop on that token.
        with torch.no_grad():
            model.lm_head.weight[7] += 100.0
        out = model.generate(
            torch.randint(0, 256, (1, 2)), max_new_tokens=50, temperature=0.0, eot_id=7
        )
        assert out.shape[1] < 52

    def test_generation_past_the_context_window_does_not_crash(self):
        model = NanoLM(tiny(block_size=16))
        out = model.generate(torch.randint(0, 256, (1, 8)), max_new_tokens=40, temperature=0.8)
        assert out.shape[1] == 48

    def test_top_k_restricts_the_sampled_set(self):
        torch.manual_seed(0)
        model = NanoLM(tiny())
        prompt = torch.randint(0, 256, (1, 4))
        sampled = {
            int(t)
            for _ in range(30)
            for t in model.generate(prompt, max_new_tokens=1, top_k=1, temperature=1.0)[0, -1:]
        }
        assert len(sampled) == 1  # top_k=1 is greedy


class TestOptimizerAndSchedule:
    def test_only_matrices_get_weight_decay(self):
        model = NanoLM(tiny())
        groups = make_optimizer(model, TrainConfig()).param_groups
        assert groups[0]["weight_decay"] > 0
        assert groups[1]["weight_decay"] == 0
        assert all(p.dim() >= 2 for p in groups[0]["params"])
        assert all(p.dim() < 2 for p in groups[1]["params"])

    def test_warmup_rises_then_cosine_decays(self):
        config = TrainConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100)
        assert lr_at(0, config) < lr_at(5, config) < lr_at(9, config)
        assert lr_at(10, config) == pytest.approx(1e-3, rel=0.02)
        assert lr_at(99, config) < lr_at(50, config) < lr_at(10, config)

    def test_lr_never_falls_below_the_floor(self):
        config = TrainConfig(learning_rate=1e-3, min_lr_ratio=0.1, warmup_steps=5, max_steps=50)
        assert lr_at(200, config) >= 1e-4 * 0.999


class TestCheckpoints:
    def test_round_trip_preserves_predictions(self, tmp_path):
        torch.manual_seed(0)
        model = NanoLM(tiny())
        model.eval()
        run = TrainingRun(model_config=model.config.to_dict(), train_config={})
        path = tmp_path / "model.pt"
        save_checkpoint(path, model, run)

        restored, meta = load_checkpoint(path)
        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            assert torch.allclose(model(x)[0], restored(x)[0], atol=1e-6)
        assert meta["model_config"]["n_layer"] == model.config.n_layer

    def test_checkpoint_records_provenance(self, tmp_path):
        """A model you cannot trace to its data is the thing this repo objects to."""
        model = NanoLM(tiny())
        run = TrainingRun(
            model_config=model.config.to_dict(),
            train_config={},
            corpus_sha256="a" * 64,
            vocab_size=256,
        )
        path = tmp_path / "model.pt"
        save_checkpoint(path, model, run)
        _, meta = load_checkpoint(path)
        assert meta["corpus_sha256"] == "a" * 64
        assert "git_commit" in meta


class TestTrainingActuallyLearns:
    def test_loss_drops_on_a_learnable_pattern(self, tmp_path):
        """A repeating sequence should be memorized fast. If this does not fall,
        the training loop is broken regardless of what the model does."""
        import numpy as np

        from punkai.nano.data import TokenData, write_token_bin
        from punkai.nano.train import train

        pattern = ([3, 1, 4, 1, 5, 9, 2, 6] * 4000)[:32000]
        path = tmp_path / "tokens.bin"
        write_token_bin(pattern, path, 256)
        assert np.fromfile(path, dtype=np.uint16).shape[0] == 32000

        data = TokenData(path, 256)
        torch.manual_seed(0)
        model = NanoLM(tiny(block_size=16))
        config = TrainConfig(
            out_dir=str(tmp_path / "run"),
            batch_size=8,
            block_size=16,
            max_steps=60,
            warmup_steps=5,
            eval_interval=30,
            log_interval=1000,
            device="cpu",
        )
        run = TrainingRun(model_config=model.config.to_dict(), train_config=config.to_dict())
        train(model, data, config, run, progress=False)

        assert run.history[0]["val"] > run.history[-1]["val"]
        assert run.final_val_loss < math.log(256) * 0.8
        assert (tmp_path / "run" / "model.pt").exists()


class TestTokenData:
    def test_validation_split_is_the_tail_not_a_random_sample(self, tmp_path):
        """Random splits of contiguous text leak neighbours across the boundary."""
        from punkai.nano.data import TokenData, write_token_bin

        path = tmp_path / "t.bin"
        write_token_bin(list(range(1000)), path, 4096)
        data = TokenData(path, 4096, val_fraction=0.1)
        assert len(data.train) == 900
        assert int(data.val[0]) == 900

    def test_batches_are_shaped_and_offset_by_one(self, tmp_path):
        from punkai.nano.data import TokenData, write_token_bin

        path = tmp_path / "t.bin"
        write_token_bin(list(range(500)) * 4, path, 4096)
        data = TokenData(path, 4096)
        x, y = data.get_batch("train", 4, 16)
        assert x.shape == (4, 16) and y.shape == (4, 16)
        assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted by one

    def test_corpus_too_small_says_so(self, tmp_path):
        from punkai.nano.data import TokenData, write_token_bin

        path = tmp_path / "t.bin"
        write_token_bin(list(range(20)), path, 4096)
        data = TokenData(path, 4096)
        with pytest.raises(ValueError, match="too few for block_size"):
            data.get_batch("train", 2, 128)

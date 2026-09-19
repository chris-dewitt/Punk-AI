"""The tokenizer is pure stdlib, so these run everywhere -- no torch needed."""

import pytest

from punkai.nano.tokenizer import ENDOFTEXT, BPETokenizer, pretokenize

CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 60
    + "you own the weights and the tokenizer too. " * 60
    + "solder copper antenna lattice drift signal static. " * 40
)


@pytest.fixture(scope="module")
def tokenizer():
    return BPETokenizer.train(CORPUS, vocab_size=512)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            "the quick brown fox",
            "unseen vocabulary appears here",
            "",
            " ",
            "\n\n\t",
            "MiXeD CaSe 12345 !@#$%",
            "café niño 日本語 🚀🎸",
            "a" * 500,
        ],
    )
    def test_decode_inverts_encode(self, tokenizer, text):
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_every_byte_is_representable(self, tokenizer):
        """Byte-level means there is no such thing as an unknown token."""
        raw = bytes(range(256)).decode("latin-1")
        assert tokenizer.decode(tokenizer.encode(raw)) == raw

    def test_untrained_tokenizer_is_still_a_valid_byte_coder(self):
        bare = BPETokenizer()
        assert bare.vocab_size == 256
        assert bare.decode(bare.encode("hello")) == "hello"


class TestTraining:
    def test_learns_merges_and_respects_vocab_budget(self, tokenizer):
        assert tokenizer.merges
        assert tokenizer.vocab_size <= 512

    def test_merges_actually_compress(self, tokenizer):
        bare = BPETokenizer()
        assert len(tokenizer.encode(CORPUS)) < len(bare.encode(CORPUS)) / 2
        assert tokenizer.compression(CORPUS) > 2.0

    def test_frequent_words_become_single_tokens(self, tokenizer):
        """The whole point of BPE: common strings collapse."""
        assert len(tokenizer.encode(" the")) == 1

    def test_bigger_vocab_compresses_better(self):
        small = BPETokenizer.train(CORPUS, vocab_size=300)
        large = BPETokenizer.train(CORPUS, vocab_size=600)
        assert large.compression(CORPUS) >= small.compression(CORPUS)

    def test_vocab_below_the_byte_floor_is_rejected(self):
        with pytest.raises(ValueError, match="below the"):
            BPETokenizer.train(CORPUS, vocab_size=100)

    def test_training_is_deterministic(self):
        first = BPETokenizer.train(CORPUS, vocab_size=400)
        second = BPETokenizer.train(CORPUS, vocab_size=400)
        assert first.merges == second.merges

    def test_records_corpus_provenance(self, tokenizer):
        """You should always be able to ask what a tokenizer was trained on."""
        assert len(tokenizer.corpus_sha256) == 64
        assert tokenizer.corpus_chars == len(CORPUS)


class TestSpecials:
    def test_eot_is_outside_the_learned_vocabulary(self, tokenizer):
        assert tokenizer.eot_id >= 256 + len(tokenizer.merges)

    def test_eot_round_trips(self, tokenizer):
        ids = tokenizer.encode("hello", add_eot=True)
        assert ids[-1] == tokenizer.eot_id
        assert tokenizer.decode(ids) == "hello" + ENDOFTEXT

    def test_text_that_looks_like_a_special_token_is_just_text(self, tokenizer):
        """A user typing the literal string must not become a control token."""
        ids = tokenizer.encode(ENDOFTEXT)
        assert tokenizer.eot_id not in ids


class TestPersistence:
    def test_save_load_round_trip(self, tokenizer, tmp_path):
        path = tmp_path / "tok.json"
        tokenizer.save(path)
        reloaded = BPETokenizer.load(path)
        assert reloaded.merges == tokenizer.merges
        assert reloaded.specials == tokenizer.specials
        assert reloaded.corpus_sha256 == tokenizer.corpus_sha256
        text = "the quick brown fox owns the weights"
        assert reloaded.encode(text) == tokenizer.encode(text)

    def test_unknown_format_version_rejected(self, tmp_path):
        path = tmp_path / "tok.json"
        path.write_text('{"version": 99, "merges": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported tokenizer format"):
            BPETokenizer.load(path)

    def test_out_of_range_token_is_an_error_not_a_crash(self, tokenizer):
        with pytest.raises(ValueError, match="outside this vocabulary"):
            tokenizer.decode([999999])

    def test_invalid_utf8_from_a_model_decodes_to_a_smudge(self, tokenizer):
        """Sampling can emit byte sequences that are not valid UTF-8."""
        assert isinstance(tokenizer.decode([0xC3]), str)


class TestPretokenizer:
    def test_splits_words_keeping_leading_space(self):
        assert pretokenize("hello world") == ["hello", " world"]

    def test_contractions_split_off(self):
        assert "'s" in pretokenize("the dog's bone")

    def test_reassembles_exactly(self):
        text = "Hello,   world!\n\tTabbed 42 times."
        assert "".join(pretokenize(text)) == text

    def test_merges_never_span_word_boundaries(self):
        """Pre-tokenizing is what stops BPE learning 'of the' as one symbol."""
        tokenizer = BPETokenizer.train("of the " * 500, vocab_size=300)
        assert len(tokenizer.encode("of the")) >= 2

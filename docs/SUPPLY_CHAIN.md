# Supply chain: how to download a model on purpose

The short version:

1. Pin the commit sha. Never a branch, never a tag.
2. Fetch `.safetensors`. Do not fetch pickles you will not use.
3. Hash what arrived. Write the hashes into a manifest and commit it.
4. Read any `.py` in the repo before it can ever run.
5. Record the licence and what it permits, in the same file.

`punk verify` is step 3 and 4 automated. The rest is judgement.

---

## Why a branch is not a pin

`Qwen/Qwen2.5-7B-Instruct` at `main` is whatever the owner pushed most recently.
The owner can force-push. The account can be compromised. The tag you pinned can
be moved to a different commit with the same name. A 40-character commit sha is
content-addressed: it is either what you reviewed or it is a different string.

```python
ModelManifest(revision="main")              # ManifestError
ModelManifest(revision="a1b2c3…40 chars")   # fine
```

This is the same argument as pinning a lockfile, and the stakes are higher,
because a model file gets loaded by a library that historically executed pickles.

## Why `.safetensors` and not `.bin`

| Format | What loading it does |
|---|---|
| `.safetensors` | Reads a JSON header and memory-maps tensors. No code path to execute. |
| `.gguf` | Same idea, llama.cpp's format. Data, not code. |
| `.bin`, `.pt`, `.pth`, `.ckpt` | Python pickle. `pickle.load` **is** an arbitrary code execution primitive. |

Since torch 2.6, `torch.load` defaults to `weights_only=True`, which blocks the
classic attack. That is a real improvement and not a reason to relax: older
torch is everywhere, `weights_only=False` is still one keyword away, and the
safe option costs you nothing.

`download_model()` excludes pickle formats from the download itself rather than
ignoring them at load time. Bytes you never fetched cannot be loaded by accident
six months later by a different script.

## `trust_remote_code`

This flag imports Python from the model repo and runs it in your process. Some
genuinely good models need it — a new architecture ships its modeling code
before `transformers` supports it natively.

The rule in this repo: allowed only when a manifest says so **and** carries a
`review_note`. Not because the note is a security control, but because it forces
the decision to be made once, in writing, by someone who read the file.

```json
{
  "allow_remote_code": true,
  "review_note": "read modeling_foo.py at this sha on 2026-03-02 (chris): pure
                  torch, no network, no subprocess, no file writes outside the
                  model dir."
}
```

If you cannot write that sentence honestly, do not set the flag.

## Building a manifest for a model you already have

```bash
python - <<'EOF'
from pathlib import Path
from punkai.registry.manifest import FileEntry, ModelManifest
from punkai.registry.verify import sha256_file

root = Path("./weights/qwen2.5-7b-instruct")
files = [
    FileEntry(p.relative_to(root).as_posix(), sha256_file(p), p.stat().st_size)
    for p in sorted(root.rglob("*")) if p.is_file()
]
ModelManifest(
    name="qwen2.5-7b-instruct",
    source="hf:Qwen/Qwen2.5-7B-Instruct",
    revision="<the 40-char commit sha you actually downloaded>",
    license="apache-2.0",
    files=files,
    params_b=7.62, num_layers=28, num_kv_heads=4, head_dim=128,
    context_length=32768,
).save("models/qwen2.5-7b-instruct.json")
EOF

punk verify models/qwen2.5-7b-instruct.json ./weights/qwen2.5-7b-instruct
```

Commit the manifest. It is now a tripwire: if those files ever change — a bad
re-download, a stomped cache, a tampered disk — `punk verify` says so in
seconds.

## A note on the Hugging Face cache

The hub cache stores blobs once and symlinks snapshot directories at them. That
means a model directory legitimately contains symlinks pointing outside itself,
which is why `scan_directory` reports those as a **warning**, not an error. In a
directory you populated yourself, a symlink pointing at `/etc/` is a very
different signal. Read the path it names.

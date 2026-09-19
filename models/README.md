# Manifests live here

One JSON file per model you have approved, committed to git. A manifest pins the
commit sha, records the sha256 of every file, and records the decisions you made
about risky knobs (`allow_remote_code`, `allow_pickle_weights`) with a written
review note.

Build one from a directory you already downloaded:

```bash
python examples/make_manifest.py ./weights/qwen2.5-7b-instruct \
    --source hf:Qwen/Qwen2.5-7B-Instruct \
    --revision <the 40-char commit sha you downloaded> \
    --license apache-2.0 \
    --out models/qwen2.5-7b-instruct.json
```

Then it is a tripwire:

```bash
punk verify models/qwen2.5-7b-instruct.json ./weights/qwen2.5-7b-instruct
```

Weights themselves are gitignored. The manifest is small, readable, and the
thing worth keeping in version control — see
[docs/SUPPLY_CHAIN.md](../docs/SUPPLY_CHAIN.md).

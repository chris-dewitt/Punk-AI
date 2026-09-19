"""`punk` -- the command line over everything in this repo.

    punk size llama3.1-8b --gpu rtx-3060-12g
    punk verify models/qwen2.5-7b.json ./weights/qwen2.5-7b
    punk license llama3.1 --intent commercial
    punk eval capability_smoke --backend llamacpp:http://127.0.0.1:8080
    punk serve --backend llamacpp --keys ~/.punk/keys.json --audit ~/.punk/audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from punkai import __version__
from punkai.errors import SmithError


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


# --- commands -------------------------------------------------------------


def cmd_size(args) -> int:
    from punkai.hardware import ARCHS, GPUS, QUANTS, estimate, max_context, recommend
    from punkai.hardware.sizing import gpu_budget

    if args.list:
        print("architectures:")
        for name, spec in sorted(ARCHS.items()):
            print(
                f"  {name:<20} {spec.params_b:>5.1f}B  "
                f"{spec.num_layers} layers  ctx {spec.context_length}"
            )
        print("\ngpus:")
        for name, vram in sorted(GPUS.items(), key=lambda kv: kv[1]):
            print(f"  {name:<20} {vram:>5.1f} GiB")
        print("\nquants (bits per weight):")
        for name, quant in QUANTS.items():
            print(f"  {name:<8} {quant.bits:>5.2f}  {quant.quality:<10} {quant.note}")
        return 0

    if not args.arch:
        return _fail("give an architecture, or --list to see them")
    budget = args.vram if args.vram else (gpu_budget(args.gpu) if args.gpu else None)

    est = estimate(args.arch, args.quant, args.ctx, args.batch, args.kv_bits)
    print(est.explain(budget))
    if budget:
        print()
        fits = recommend(args.arch, budget, args.ctx, args.batch)
        if fits:
            best = fits[0]
            print(f"best quant that fits at ctx {args.ctx}: {best.quant} "
                  f"({QUANTS[best.quant].quality})")
            longest = max_context(args.arch, budget, args.quant)
            print(f"max context at {args.quant}: {longest} tokens")
        else:
            print(f"nothing fits in {budget:.1f} GiB at ctx {args.ctx}. Try a shorter context, "
                  "an 8-bit KV cache (--kv-bits 8), or a smaller model.")
    return 0


def cmd_verify(args) -> int:
    from punkai.registry import ModelManifest, verify_directory

    manifest = ModelManifest.load(args.manifest)
    report = verify_directory(manifest, args.directory, allow_extra=not args.strict)
    print(report.summary())
    return 0 if report.ok else 2


def cmd_scan(args) -> int:
    from punkai.registry.verify import scan_directory

    findings = scan_directory(args.directory)
    if not findings:
        print(f"{args.directory}: nothing alarming -- no pickle weights, no Python, no binaries")
        return 0
    for finding in findings:
        print(finding)
    errors = sum(1 for f in findings if f.level == "error")
    print(f"\n{len(findings)} finding(s), {errors} error(s)")
    return 2 if errors else 0


def cmd_license(args) -> int:
    from punkai.registry import describe, gate

    if args.list:
        from punkai.registry.licenses import LICENSES

        for spec in LICENSES.values():
            print(f"  {spec.id:<20} {spec.family:<24} {spec.name}")
        return 0
    if not args.license_id:
        return _fail("give a licence id, or --list")
    decision = gate(args.license_id, args.intent)
    print(decision)
    spec = describe(args.license_id)
    if spec.id == "unknown" and args.license_id.lower() not in {"unknown", "other", ""}:
        print(f"\nnote: {args.license_id!r} is not in the table; treated as unknown.")
    return 0 if decision.allowed else 2


def cmd_eval(args) -> int:
    from punkai.evals import Suite, compare, run_suite
    from punkai.serve.backends import load_backend

    if args.list:
        for name in Suite.list_builtin():
            suite = Suite.builtin(name)
            print(f"  {name:<24} {len(suite.cases):>3} cases  {suite.description[:60]}")
        return 0
    if not args.suite:
        return _fail("give a suite name or path, or --list")

    suite = Suite.load(args.suite) if Path(args.suite).exists() else Suite.builtin(args.suite)
    backend = load_backend(args.backend)
    result = run_suite(
        suite,
        lambda prompt: backend.generate(prompt, args.max_tokens, args.temperature),
        model_name=args.model or backend.name,
        meta={"backend": args.backend, "temperature": args.temperature},
    )
    print(result.summary(show_failures=args.show_failures))
    if args.save:
        result.save(args.save)
        print(f"\nsaved to {args.save}")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        comparison = compare(baseline, result)
        print()
        print(comparison.summary())
        if not comparison.safe_to_ship:
            return 2
    return 0


def cmd_guard(args) -> int:
    from punkai.guards import GuardPolicy

    policy = {"strict": GuardPolicy.strict, "permissive": GuardPolicy.permissive,
              "default": GuardPolicy}[args.policy]()
    text = args.text if args.text else sys.stdin.read()
    result = policy.check_input(text)
    print(f"injection score : {result.injection_score:.2f}")
    print(f"blocked         : {result.blocked}")
    for signal in result.signals:
        print(f"  signal {signal.name} ({signal.weight}): {signal.excerpt!r}")
    for reason in result.reasons:
        print(f"  reason: {reason}")
    if result.text != text:
        print(f"\ncleaned text:\n{result.text}")
    return 2 if result.blocked else 0


def cmd_keys(args) -> int:
    from punkai.serve.auth import KeyStore

    store = KeyStore(args.store)
    if args.keys_command == "issue":
        key, record = store.issue(args.label, rate_per_minute=args.rate)
        print("Key issued. This is the only time it will be shown:\n")
        print(f"  {key}\n")
        print(f"  id    : {record.key_id}")
        print(f"  label : {record.label}")
        print(f"  rate  : {record.rate_per_minute}/min")
        print(f"  store : {args.store}")
        return 0
    if args.keys_command == "list":
        if not store.keys:
            print(f"no keys in {args.store}")
            return 0
        for record in store.keys.values():
            state = "active" if record.active else "REVOKED"
            print(f"  {record.key_id}  {state:<8} {record.rate_per_minute:>4}/min  {record.label}")
        return 0
    if args.keys_command == "revoke":
        if store.revoke(args.key_id):
            print(f"revoked {args.key_id}")
            return 0
        return _fail(f"no active key with id {args.key_id}")
    return _fail("use: punk keys issue|list|revoke")


def cmd_audit(args) -> int:
    from punkai.serve.audit import verify_chain

    check = verify_chain(args.logfile)
    if check.ok:
        print(f"chain intact: {check.count} record(s)")
        print(f"head hash   : {check.head_hash}")
        print("\nRecord that head hash somewhere off this machine -- the chain proves nothing")
        print("was edited, but only an external anchor proves nothing was cut off the end.")
        return 0
    print(f"CHAIN BROKEN after {check.count} record(s):")
    for problem in check.problems:
        print(f"  {problem}")
    return 2


def cmd_data(args) -> int:
    from punkai.train.data import Example, audit_dataset

    examples = []
    with open(args.dataset, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                return _fail(f"{args.dataset}:{lineno}: {exc}")
            examples.append(
                Example(
                    prompt=str(row.get("prompt", row.get("instruction", ""))),
                    response=str(row.get("response", row.get("output", ""))),
                    source=str(row.get("source", "unknown")),
                    license=str(row.get("license", "unknown")),
                )
            )
    eval_texts = None
    if args.against:
        eval_texts = [
            json.loads(line)["prompt"]
            for line in Path(args.against).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    report = audit_dataset(examples, eval_texts)
    print(report.summary())
    if args.card:
        Path(args.card).write_text(report.card(Path(args.dataset).stem) + "\n", encoding="utf-8")
        print(f"\ndataset card written to {args.card}")
    return 0 if report.clean else 2


def cmd_plan(args) -> int:
    from punkai.hardware.sizing import gpu_budget
    from punkai.train import LoraPlan, plan_summary

    budget = args.vram if args.vram else (gpu_budget(args.gpu) if args.gpu else None)
    plan = LoraPlan(
        arch=args.arch,
        rank=args.rank,
        seq_len=args.seq,
        batch_size=args.batch,
        grad_checkpointing=not args.no_checkpointing,
    )
    print(plan_summary(plan, budget))
    return 0


def cmd_serve(args) -> int:
    from punkai.guards import GuardPolicy
    from punkai.serve import AuditLog, KeyStore, ServerConfig, load_backend, serve_forever

    policy = {"strict": GuardPolicy.strict, "permissive": GuardPolicy.permissive,
              "default": GuardPolicy}[args.policy]()
    config = ServerConfig(
        backend=load_backend(args.backend),
        policy=policy,
        keystore=KeyStore(args.keys) if args.keys else KeyStore(),
        audit=AuditLog(args.audit, log_content=args.log_content),
        model_name=args.model,
        host=args.host,
        port=args.port,
        require_auth=not args.no_auth,
    )
    serve_forever(config)
    return 0


# --- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="punk",
        description="Own the weights, own the stack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"punk-ai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("size", help="will this model fit on my card?")
    p.add_argument("arch", nargs="?")
    p.add_argument("--gpu", help="a known gpu name (see --list)")
    p.add_argument("--vram", type=float, help="VRAM budget in GiB, overrides --gpu")
    p.add_argument("--quant", default="q4_k_m")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--kv-bits", type=int, default=16, dest="kv_bits", choices=[8, 16])
    p.add_argument("--list", action="store_true", help="list known archs, gpus and quants")
    p.set_defaults(func=cmd_size)

    p = sub.add_parser("verify", help="hash a model dir against its manifest")
    p.add_argument("manifest")
    p.add_argument("directory")
    p.add_argument("--strict", action="store_true", help="unexpected files are errors")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("scan", help="look for pickles, scripts and binaries in a model dir")
    p.add_argument("directory")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("license", help="what does this licence let me do?")
    p.add_argument("license_id", nargs="?")
    p.add_argument(
        "--intent",
        default="local-tinkering",
        choices=[
            "local-tinkering", "research", "commercial",
            "redistribute", "finetune", "distill",
        ],
    )
    p.add_argument("--list", action="store_true")
    p.set_defaults(func=cmd_license)

    p = sub.add_parser("eval", help="run an eval suite against a backend")
    p.add_argument("suite", nargs="?")
    p.add_argument("--backend", default="echo")
    p.add_argument("--model", default="")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--save", help="write results JSON here")
    p.add_argument("--baseline", help="compare against a saved results JSON")
    p.add_argument("--show-failures", type=int, default=5)
    p.add_argument("--list", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("guard", help="run the input guards over some text")
    p.add_argument("text", nargs="?", help="text to check, or pipe it on stdin")
    p.add_argument("--policy", default="default", choices=["default", "strict", "permissive"])
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("keys", help="issue, list and revoke API keys")
    p.add_argument("--store", default=str(Path.home() / ".punk" / "keys.json"))
    keys_sub = p.add_subparsers(dest="keys_command", required=True)
    k = keys_sub.add_parser("issue")
    k.add_argument("--label", required=True)
    k.add_argument("--rate", type=int, default=60, help="requests per minute")
    keys_sub.add_parser("list")
    k = keys_sub.add_parser("revoke")
    k.add_argument("key_id")
    p.set_defaults(func=cmd_keys)

    p = sub.add_parser("audit", help="verify an audit chain")
    p.add_argument("logfile")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("data", help="audit a jsonl fine-tuning dataset")
    p.add_argument("dataset")
    p.add_argument("--against", help="eval jsonl to check for contamination")
    p.add_argument("--card", help="write a dataset card here")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("plan", help="plan a QLoRA run and check it fits")
    p.add_argument("arch")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--gpu")
    p.add_argument("--vram", type=float)
    p.add_argument("--no-checkpointing", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("serve", help="serve a model behind auth, limits and guards")
    p.add_argument("--backend", default="echo", help="echo | llamacpp:URL | transformers:DIR")
    p.add_argument("--model", default="local")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--keys", default=str(Path.home() / ".punk" / "keys.json"))
    p.add_argument("--audit", default=str(Path.home() / ".punk" / "audit.jsonl"))
    p.add_argument("--policy", default="default", choices=["default", "strict", "permissive"])
    p.add_argument(
        "--log-content", action="store_true", help="store prompt text in the audit log"
    )
    p.add_argument("--no-auth", action="store_true", help="loopback only; refuses a public bind")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SmithError as exc:
        return _fail(str(exc))
    except (KeyError, ValueError) as exc:
        return _fail(str(exc))
    except FileNotFoundError as exc:
        return _fail(f"{exc.filename}: no such file")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

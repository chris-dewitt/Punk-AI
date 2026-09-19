"""What the licence on those weights actually lets you do.

"Open weights" is a marketing phrase, not a legal one. The spread between
Apache-2.0 and "you may not use the outputs to train another model, and if you
cross 700M monthly users you need a separate agreement" is enormous, and you
want to know which one you are standing on *before* you build on it.

This module encodes the obligations that bite in practice. It is a lookup
table maintained by a human, not legal advice, and licences get revised --
`describe()` always hands back the upstream URL so you can read the text
yourself. Treat a `needs_review` result as "go read it", not "you're fine".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Answer = Literal["yes", "no", "conditional", "unknown"]

Intent = Literal[
    "local-tinkering",  # run it on your own box, outputs stay with you
    "research",  # publish findings, non-commercial
    "commercial",  # ship it in something that makes money
    "redistribute",  # rehost the weights, or ship them inside a product
    "finetune",  # train a derivative and keep it
    "distill",  # use its outputs as training data for another model
]


@dataclass(frozen=True)
class LicenseSpec:
    id: str
    name: str
    family: str
    commercial: Answer = "unknown"
    redistribute: Answer = "unknown"
    finetune: Answer = "unknown"
    distill: Answer = "unknown"
    obligations: tuple[str, ...] = ()
    url: str = ""
    note: str = ""


def _spec(**kw) -> LicenseSpec:
    return LicenseSpec(**kw)


# Ordered roughly from "do what you like" to "read every word twice".
LICENSES: dict[str, LicenseSpec] = {
    spec.id: spec
    for spec in [
        _spec(
            id="apache-2.0",
            name="Apache License 2.0",
            family="permissive",
            commercial="yes",
            redistribute="yes",
            finetune="yes",
            distill="yes",
            obligations=(
                "Keep the LICENSE and NOTICE files with any redistribution.",
                "State significant changes you made.",
            ),
            url="https://www.apache.org/licenses/LICENSE-2.0",
            note="The friendliest thing an open-weight model can carry. Qwen2.5, Mistral 7B "
            "v0.3, OLMo, SmolLM and many others ship under it.",
        ),
        _spec(
            id="mit",
            name="MIT License",
            family="permissive",
            commercial="yes",
            redistribute="yes",
            finetune="yes",
            distill="yes",
            obligations=("Keep the copyright notice with any redistribution.",),
            url="https://opensource.org/license/mit",
        ),
        _spec(
            id="bsd-3-clause",
            name="BSD 3-Clause",
            family="permissive",
            commercial="yes",
            redistribute="yes",
            finetune="yes",
            distill="yes",
            obligations=("Keep the notice; do not use the authors' names to endorse.",),
            url="https://opensource.org/license/bsd-3-clause",
        ),
        _spec(
            id="llama3-community",
            name="Llama 3.x Community License",
            family="open-weight-conditional",
            commercial="conditional",
            redistribute="conditional",
            finetune="yes",
            distill="conditional",
            obligations=(
                'Display "Built with Llama" where you use it.',
                'Name any derivative model with "Llama-" at the front.',
                "Ship the licence text and the Acceptable Use Policy with redistributions.",
                "Over 700M monthly active users at the time of release, you must request a "
                "separate licence from Meta.",
                "Check the version you are on: the 3.1+ text permits using outputs to improve "
                "other models, earlier Llama terms did not.",
            ),
            url="https://www.llama.com/llama3_1/license/",
            note="Not an OSI-approved open source licence, despite how it is usually described.",
        ),
        _spec(
            id="gemma",
            name="Gemma Terms of Use",
            family="open-weight-conditional",
            commercial="yes",
            redistribute="conditional",
            finetune="yes",
            distill="yes",
            obligations=(
                "Pass the same use restrictions on to anyone you give the weights to.",
                "Include the Gemma Prohibited Use Policy with redistributions.",
                "Google can require you to stop distributing if you breach the use policy.",
            ),
            url="https://ai.google.dev/gemma/terms",
        ),
        _spec(
            id="mistral-research",
            name="Mistral AI Research License",
            family="non-commercial",
            commercial="no",
            redistribute="conditional",
            finetune="conditional",
            distill="no",
            obligations=(
                "Research and personal use only -- a commercial deployment needs a paid licence.",
                "Derivatives inherit the same restriction.",
            ),
            url="https://mistral.ai/licenses/MRL-0.1.md",
            note="Several of the larger Mistral models are MRL, not Apache. Check per model.",
        ),
        _spec(
            id="cc-by-nc-4.0",
            name="Creative Commons BY-NC 4.0",
            family="non-commercial",
            commercial="no",
            redistribute="yes",
            finetune="yes",
            distill="conditional",
            obligations=(
                "Attribute the original.",
                "No commercial use, and 'we only charge for hosting' is still commercial use.",
            ),
            url="https://creativecommons.org/licenses/by-nc/4.0/",
        ),
        _spec(
            id="cc-by-sa-4.0",
            name="Creative Commons BY-SA 4.0",
            family="copyleft",
            commercial="yes",
            redistribute="yes",
            finetune="conditional",
            distill="conditional",
            obligations=(
                "Attribute the original.",
                "Share derivatives under the same licence -- this can reach your fine-tune.",
            ),
            url="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        _spec(
            id="openrail-m",
            name="BigScience OpenRAIL-M",
            family="responsible-ai",
            commercial="yes",
            redistribute="yes",
            finetune="yes",
            distill="yes",
            obligations=(
                "The use restrictions in the annex travel with the weights and every derivative.",
                "You must pass the restrictions on to downstream users.",
            ),
            url="https://www.licenses.ai/",
            note="Commercially usable, but behaviourally restricted -- it is not OSI open source.",
        ),
        _spec(
            id="unknown",
            name="No licence stated",
            family="unknown",
            commercial="unknown",
            redistribute="unknown",
            finetune="unknown",
            distill="unknown",
            obligations=(
                "Weights with no licence are not permissively licensed by default -- the safe "
                "reading is that you have no grant at all.",
            ),
            note="Find the licence before you build anything on it.",
        ),
    ]
}

# Common spellings that show up in model cards and HF metadata.
ALIASES: dict[str, str] = {
    "apache2": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache 2.0": "apache-2.0",
    "llama3": "llama3-community",
    "llama3.1": "llama3-community",
    "llama3.2": "llama3-community",
    "llama3.3": "llama3-community",
    "llama-3.1": "llama3-community",
    "llama4": "llama3-community",
    "gemma-terms-of-use": "gemma",
    "gemma2": "gemma",
    "mrl": "mistral-research",
    "mistral-ai-research": "mistral-research",
    "cc-by-nc": "cc-by-nc-4.0",
    "cc-by-sa": "cc-by-sa-4.0",
    "openrail": "openrail-m",
    "creativeml-openrail-m": "openrail-m",
    "other": "unknown",
    "": "unknown",
}

_INTENT_FIELD: dict[Intent, str] = {
    "local-tinkering": "",  # no licence in the table forbids running it yourself
    "research": "",
    "commercial": "commercial",
    "redistribute": "redistribute",
    "finetune": "finetune",
    "distill": "distill",
}


@dataclass
class LicenseDecision:
    license_id: str
    intent: Intent
    allowed: bool
    needs_review: bool
    reasons: list[str] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)
    url: str = ""

    def __str__(self) -> str:
        verdict = "OK" if self.allowed and not self.needs_review else (
            "REVIEW" if self.allowed else "BLOCKED"
        )
        lines = [f"[{verdict}] {self.license_id} for intent '{self.intent}'"]
        lines += [f"  - {r}" for r in self.reasons]
        if self.obligations:
            lines.append("  obligations:")
            lines += [f"    * {o}" for o in self.obligations]
        if self.url:
            lines.append(f"  read it: {self.url}")
        return "\n".join(lines)


def normalize(license_id: str) -> str:
    key = (license_id or "").strip().lower()
    key = ALIASES.get(key, key)
    return key if key in LICENSES else "unknown"


def describe(license_id: str) -> LicenseSpec:
    """Look up a licence, tolerating the spellings model cards actually use."""
    return LICENSES[normalize(license_id)]


def gate(license_id: str, intent: Intent = "local-tinkering") -> LicenseDecision:
    """Decide whether `intent` is permitted, and what you owe if it is.

    `allowed=False` means the licence says no. `needs_review=True` means the
    answer is conditional or unknown and a human has to read the text. Neither
    is legal advice; both are better than finding out in a deposition.
    """
    spec = describe(license_id)
    if intent not in _INTENT_FIELD:
        raise ValueError(f"unknown intent {intent!r}; expected one of {sorted(_INTENT_FIELD)}")

    decision = LicenseDecision(
        license_id=spec.id,
        intent=intent,
        allowed=True,
        needs_review=False,
        obligations=list(spec.obligations),
        url=spec.url,
    )

    field_name = _INTENT_FIELD[intent]
    if not field_name:
        if spec.id == "unknown":
            decision.needs_review = True
            decision.reasons.append(
                "No licence on record. Running it locally is low risk in practice, but you "
                "have no grant in writing -- find the licence before it leaves your machine."
            )
        else:
            decision.reasons.append(f"{spec.name} does not restrict local, private use.")
        return decision

    answer: Answer = getattr(spec, field_name)
    if answer == "yes":
        decision.reasons.append(f"{spec.name} permits {intent}.")
    elif answer == "no":
        decision.allowed = False
        decision.reasons.append(f"{spec.name} prohibits {intent}.")
    elif answer == "conditional":
        decision.needs_review = True
        decision.reasons.append(
            f"{spec.name} permits {intent} only under conditions -- read the obligations below."
        )
    else:
        decision.needs_review = True
        decision.allowed = spec.id != "unknown"
        decision.reasons.append(f"{spec.name}: no recorded answer for {intent}. Read the text.")

    if spec.note:
        decision.reasons.append(spec.note)
    return decision

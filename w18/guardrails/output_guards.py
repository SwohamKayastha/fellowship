"""Output guardrails: schema validation, groundedness, sensitive-information detection."""
import json
import re
from dataclasses import dataclass, field

from .pii import detect_pii, mask_pii
from .reference import CATEGORIES, REFERENCE

# ---- Schema validation ----------------------------------------------------

REQUIRED_FIELDS = {"answer": str, "confidence": float, "sources": list, "category": str}


@dataclass
class SchemaResult:
    valid: bool
    data: dict | None = None
    errors: list[str] = field(default_factory=list)


def validate_schema(raw: str) -> SchemaResult:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return SchemaResult(False, errors=[f"not valid JSON: {e.msg}"])
    if not isinstance(data, dict):
        return SchemaResult(False, errors=["top-level value must be an object"])

    errors = []
    for name, typ in REQUIRED_FIELDS.items():
        if name not in data:
            errors.append(f"missing field: {name}")
        elif typ is float:
            v = data[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
                errors.append("confidence must be a number in [0, 1]")
        elif not isinstance(data[name], typ):
            errors.append(f"{name} must be {typ.__name__}")
    errors += [f"unexpected field: {k}" for k in data if k not in REQUIRED_FIELDS]
    if isinstance(data.get("answer"), str) and not 0 < len(data["answer"].strip()) <= 1000:
        errors.append("answer must be 1-1000 characters")
    if isinstance(data.get("category"), str) and data["category"] not in CATEGORIES:
        errors.append(f"category must be one of {CATEGORIES}")
    if isinstance(data.get("sources"), list):
        bad = [s for s in data["sources"] if s not in REFERENCE]
        if bad:
            errors.append(f"unknown sources: {bad}")
    return SchemaResult(not errors, data if not errors else None, errors)


# ---- Groundedness ---------------------------------------------------------

STOPWORDS = set(
    "a an the is are was were be been to of in on at for and or but not no with by from as it its this that "
    "you your we our can will may do does i they their there has have had".split()
)
SUPPORT_THRESHOLD = 0.6
ABSTENTION = re.compile(r"\b(don't|do not) have (that |any )?information\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _stem(w: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9$%/.-]*[a-z0-9]", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


@dataclass
class GroundednessResult:
    supported: list[str]
    unsupported: list[str]

    @property
    def flagged(self) -> bool:
        return bool(self.unsupported)


def check_groundedness(answer: str, reference: dict[str, str]) -> GroundednessResult:
    """A claim (sentence) is supported if enough of its content words, and all of its numbers, appear in one reference sentence."""
    ref_sentences = [s for text in reference.values() for s in _sentences(text)]
    ref_tokens = [(_content_tokens(s), _numbers(s)) for s in ref_sentences]
    supported, unsupported = [], []
    for claim in _sentences(answer):
        tokens, nums = _content_tokens(claim), _numbers(claim)
        ok = bool(ABSTENTION.search(claim))  # an honest "I don't know" makes no factual claim
        for r_tokens, r_nums in [] if ok else ref_tokens:
            overlap = len(tokens & r_tokens) / len(tokens) if tokens else 1.0
            if overlap >= SUPPORT_THRESHOLD and nums <= r_nums:
                ok = True
                break
        (supported if ok else unsupported).append(claim)
    return GroundednessResult(supported, unsupported)


# ---- Sensitive-information detection ---------------------------------------

SECRET_PATTERNS = {
    "api_key": r"\bsk-[A-Za-z0-9]{16,}\b|\bAKIA[0-9A-Z]{16}\b",
    "credential": r"\b(password|passwd|secret)\s*[:=]\s*\S+",
    "prompt_leak": r"\bsystem prompt\b|\bCONFIDENTIAL\b|\bmy instructions (are|say)\b",
}
_SECRETS = {k: re.compile(v, re.I) for k, v in SECRET_PATTERNS.items()}


@dataclass
class SensitiveResult:
    pii_kinds: list[str]
    secret_kinds: list[str]
    redacted: str

    @property
    def flagged(self) -> bool:
        return bool(self.pii_kinds or self.secret_kinds)


def check_sensitive(text: str, allow_domains=()) -> SensitiveResult:
    pii = detect_pii(text, allow_domains)
    secrets = [name for name, rx in _SECRETS.items() if rx.search(text)]
    return SensitiveResult(sorted({f.kind for f in pii}), secrets, mask_pii(text, pii))

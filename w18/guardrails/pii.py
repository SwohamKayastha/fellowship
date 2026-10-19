"""PII detection shared by the input guardrail (mask before the LLM) and output guardrail (redact)."""
import re
from dataclasses import dataclass

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){12,15}\d\b")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


@dataclass
class Finding:
    kind: str
    start: int
    end: int
    text: str


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def detect_pii(text: str, allow_domains=()) -> list[Finding]:
    """Find emails, SSNs, credit cards and phone numbers. Emails on allow_domains are public and kept."""
    candidates = []
    for m in EMAIL.finditer(text):
        if m.group().split("@")[-1].lower() not in allow_domains:
            candidates.append(Finding("EMAIL", m.start(), m.end(), m.group()))
    for m in SSN.finditer(text):
        candidates.append(Finding("SSN", m.start(), m.end(), m.group()))
    for m in CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 16 and _luhn(digits):
            candidates.append(Finding("CARD", m.start(), m.end(), m.group()))
    for m in PHONE.finditer(text):
        if 9 <= len(re.sub(r"\D", "", m.group())) <= 15:
            candidates.append(Finding("PHONE", m.start(), m.end(), m.group()))

    findings = []
    for f in candidates:  # earlier kinds win overlaps (card/SSN before phone)
        if all(f.end <= g.start or f.start >= g.end for g in findings):
            findings.append(f)
    return sorted(findings, key=lambda f: f.start)


def mask_pii(text: str, findings: list[Finding]) -> str:
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        text = text[: f.start] + f"[{f.kind}]" + text[f.end :]
    return text

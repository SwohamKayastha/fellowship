"""Input guardrails: PII, prompt injection, sensitive-topic classification."""
import re
from dataclasses import dataclass, field

from .pii import Finding, detect_pii, mask_pii


@dataclass
class PIIResult:
    flagged: bool
    findings: list[Finding]
    masked: str


def check_pii(text: str, allow_domains=()) -> PIIResult:
    findings = detect_pii(text, allow_domains)
    return PIIResult(bool(findings), findings, mask_pii(text, findings))


INJECTION_PATTERNS = {
    "override_instructions": r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|any|your)\b.{0,20}\b(instructions?|prompts?|rules|guidelines)\b",
    "reveal_prompt": r"\b(reveal|show|print|repeat|display|leak)\b.{0,30}\b(system prompt|hidden prompt|initial prompt|your instructions)\b",
    "persona_swap": r"\byou are now\b|\bpretend (to be|you are)\b|\bact as (?!a customer)",
    "jailbreak_terms": r"\bdeveloper mode\b|\bDAN\b|\bjailbreak",
    "bypass_safety": r"\bbypass\b.{0,20}\b(filters?|safety|restrictions?|guardrails?)\b",
    "secret_extraction": r"\b(reveal|show|print|give|tell)\b.{0,30}\b(api keys?|credentials|admin password|secrets?)\b",
    "chat_markup": r"</?(system|assistant)>|\[/?INST\]|^#{2,}\s*system",
}
_COMPILED = {k: re.compile(v, re.I | re.M) for k, v in INJECTION_PATTERNS.items()}


@dataclass
class InjectionResult:
    flagged: bool
    matches: list[str] = field(default_factory=list)


def check_injection(text: str) -> InjectionResult:
    matches = [name for name, rx in _COMPILED.items() if rx.search(text)]
    return InjectionResult(bool(matches), matches)


# Order is priority: the first matching category wins.
TOPICS = {
    "self_harm": (r"\bkill myself\b|\bsuicid|\bend my life\b|\bself[- ]harm\b|\bhurt myself\b", "escalate",
                  "I'm really sorry you're going through this. You deserve support from a person right now. "
                  "Please contact a local emergency number or a crisis line. I can't help with this in this chat."),
    "violence": (r"\b(build|make|create)\b.{0,20}\b(bomb|weapon|explosive)s?\b", "refuse",
                 "I can't help with that request."),
    "medical": (r"\bdosage\b|\bdiagnos|\bprescri|\bsymptoms?\b|\bmedication\b", "refuse",
                "I can't give medical advice. Please talk to a qualified health professional."),
    "financial_advice": (r"\binvest(ing|ment)?\b|\bstock tips?\b|\bshould i buy\b", "refuse",
                         "I can't give investment advice. I can only answer questions about Acme Cloud's policies."),
    "legal": (r"\bsue\b|\blawsuit\b|\blawyer\b|\battorney\b|\blegal advice\b", "disclaimer",
              "Note: this is general policy information, not legal advice."),
}
_TOPICS = {k: (re.compile(p, re.I), a, m) for k, (p, a, m) in TOPICS.items()}


@dataclass
class TopicResult:
    category: str | None
    action: str | None = None  # escalate | refuse | disclaimer
    message: str = ""


def classify_topic(text: str) -> TopicResult:
    for category, (rx, action, message) in _TOPICS.items():
        if rx.search(text):
            return TopicResult(category, action, message)
    return TopicResult(None)

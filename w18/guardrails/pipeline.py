"""Pipeline: input guardrails -> LLM -> output guardrails. Every decision is recorded in `trace`."""
import re
from dataclasses import dataclass, field

from .input_guards import check_injection, check_pii, classify_topic
from .llm import get_llm
from .output_guards import check_groundedness, check_sensitive, validate_schema
from .reference import REFERENCE

MAX_ATTEMPTS = 2
FALLBACK = "I'm sorry, I couldn't produce a reliable answer. Please contact support@acmecloud.example."
BLOCKED_INJECTION = "I can't help with that request."
BLOCKED_SENSITIVE = "I'm sorry, I can't share that information."

# Emails that appear in the reference are public contact points, not leaked PII.
ALLOW_DOMAINS = {
    d.lower() for text in REFERENCE.values() for d in re.findall(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
}


@dataclass
class Result:
    status: str  # ok | partial | blocked_injection | escalated | refused_topic | fallback | blocked_sensitive
    response: str
    trace: list[str] = field(default_factory=list)


def run(user_input: str, llm=None, reference: dict[str, str] = REFERENCE) -> Result:
    llm = llm or get_llm()
    trace = []

    pii = check_pii(user_input)
    if pii.flagged:
        trace.append(f"input.pii: masked {sorted({f.kind for f in pii.findings})}")
    question = pii.masked

    inj = check_injection(question)
    if inj.flagged:
        trace.append(f"input.injection: blocked {inj.matches}")
        return Result("blocked_injection", BLOCKED_INJECTION, trace)

    topic = classify_topic(question)
    disclaimer = ""
    if topic.category:
        trace.append(f"input.topic: {topic.category} -> {topic.action}")
        if topic.action == "escalate":
            return Result("escalated", topic.message, trace)
        if topic.action == "refuse":
            return Result("refused_topic", topic.message, trace)
        disclaimer = topic.message

    for attempt in range(MAX_ATTEMPTS):
        raw = llm(question, reference, attempt)
        schema = validate_schema(raw)
        if not schema.valid:
            trace.append(f"output.schema[{attempt}]: invalid {schema.errors}")
            continue

        # Sensitive check runs first so a secret in an unsupported sentence still blocks the response.
        sens = check_sensitive(schema.data["answer"], ALLOW_DOMAINS)
        if sens.secret_kinds:
            trace.append(f"output.sensitive: blocked {sens.secret_kinds}")
            return Result("blocked_sensitive", BLOCKED_SENSITIVE, trace)
        if sens.pii_kinds:
            trace.append(f"output.sensitive: redacted {sens.pii_kinds}")

        grounded = check_groundedness(sens.redacted, reference)
        if not grounded.supported:
            trace.append(f"output.groundedness[{attempt}]: no supported claims")
            continue
        partial = grounded.flagged
        if partial:
            trace.append(f"output.groundedness[{attempt}]: removed {len(grounded.unsupported)} unsupported claim(s)")
        answer = " ".join(grounded.supported)

        if disclaimer:
            answer = f"{answer}\n\n{disclaimer}"
        return Result("partial" if partial else "ok", answer, trace)

    trace.append("output: attempts exhausted, safe fallback")
    return Result("fallback", FALLBACK, trace)

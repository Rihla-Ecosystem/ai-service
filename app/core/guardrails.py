import re
import structlog
from typing import Optional

logger = structlog.get_logger()

MILITARY_KEYWORDS = [
    r"\bmilitary\b", r"\barmy\b", r"\bnaval\b", r"\bair force\b",
    r"\bmissile\b", r"\bweapon\b", r"\barms\b", r"\bammunition\b",
    r"\bbarracks\b", r"\bintelligence\b.*\bheadquarters\b",
    r"\bspecial forces\b", r"\bsecurity zone\b", r"\brestricted area\b",
    r"\bmilitary base\b", r"\bsoldier\b.*\bpost\b",
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{16}\b",              # credit card (raw digits)
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # credit card formatted
    r"\b[A-Z]{2}\d{6,9}\b",    # passport number pattern
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules|directions)",
    r"you\s+are\s+(now|no longer)\s+",
    r"system\s+prompt",
    r"forget\s+(everything|all|previous)",
    r"new\s+instruction",
    r"override\s+(your|all|previous)",
    r"you\s+are\s+an?\s+ai\s+(model|assistant)\s+that",
    r"dan|do\s+anything\s+now",
]


def check_military_content(text: str) -> Optional[str]:
    for pattern in MILITARY_KEYWORDS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            logger.warning("Military keyword detected", keyword=match.group(), context=text[:100])
            return match.group()
    return None


def check_pii(text: str) -> Optional[str]:
    for pattern in PII_PATTERNS:
        match = re.search(pattern, text)
        if match:
            logger.warning("PII pattern detected", pattern=pattern, context=text[:50])
            return match.group()
    return None


def check_prompt_injection(text: str) -> Optional[str]:
    for pattern in PROMPT_INJECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            logger.warning("Prompt injection attempt detected", keyword=match.group())
            return match.group()
    return None


def sanitize_output(text: str) -> str:
    if not text:
        return text
    for pattern in PII_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


class InputGuardResult:
    def __init__(self):
        self.blocked = False
        self.reason: Optional[str] = None
        self.match: Optional[str] = None

    def __repr__(self):
        return f"InputGuardResult(blocked={self.blocked}, reason={self.reason})"


class OutputGuardResult:
    def __init__(self):
        self.modified = False
        self.requires_regeneration = False
        self.reason: Optional[str] = None


def check_input(user_message: str) -> InputGuardResult:
    result = InputGuardResult()

    injection = check_prompt_injection(user_message)
    if injection:
        result.blocked = True
        result.reason = "prompt_injection_attempt"
        result.match = injection
        return result

    military = check_military_content(user_message)
    if military:
        result.blocked = True
        result.reason = "restricted_content_request"
        result.match = military
        return result

    pii = check_pii(user_message)
    if pii:
        result.blocked = True
        result.reason = "pii_detected"
        result.match = pii
        return result

    return result


def check_output(ai_response: str) -> OutputGuardResult:
    result = OutputGuardResult()

    military = check_military_content(ai_response)
    if military:
        result.requires_regeneration = True
        result.reason = "military_content_generated"
        result.modified = True
        return result

    sanitized = sanitize_output(ai_response)
    if sanitized != ai_response:
        result.modified = True
        result.reason = "pii_redacted"

    return result

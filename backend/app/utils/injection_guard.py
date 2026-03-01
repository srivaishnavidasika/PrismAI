def detect_prompt_injection(code: str) -> bool:
    suspicious_patterns = [
        "ignore previous instructions",
        "system prompt",
        "act as",
        "jailbreak"
    ]

    lower_code = code.lower()

    for pattern in suspicious_patterns:
        if pattern in lower_code:
            return True

    return False
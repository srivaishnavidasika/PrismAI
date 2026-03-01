def validate_code_input(code: str) -> bool:
    if not code or len(code.strip()) == 0:
        return False

    if len(code) > 5000:
        return False

    return True
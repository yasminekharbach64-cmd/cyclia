def classify_question(question: str) -> str:
    if "diabetes" in question.lower():
        return "health"
    return "general"

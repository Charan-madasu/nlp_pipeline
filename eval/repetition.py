from collections import Counter


def repetition_rate(text: str, n: int = 2) -> float:
    """Measure n-gram repetition. Higher = more repetitive."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - (unique / total)


def avg_repetition(texts: list, n: int = 2) -> float:
    """Average repetition rate across multiple texts."""
    scores = [repetition_rate(t, n) for t in texts]
    return sum(scores) / len(scores)

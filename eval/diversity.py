from collections import Counter


def distinct_n(texts: list, n: int) -> float:
    """
    Distinct-N metric.
    Ratio of unique n-grams to total n-grams across all texts.
    Higher = more diverse output.
    """
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total


def diversity_report(texts: list) -> dict:
    """Return Distinct-1 and Distinct-2 scores."""
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
    }

import json
import numpy as np
from itertools import combinations


def load_embeddings(path: str) -> dict:
    """Load embeddings from JSON file."""
    with open(path, "r") as f:
        embeddings = json.load(f)
    # Convert lists back to numpy arrays
    return {word: np.array(vec) for word, vec in embeddings.items()}


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / (norm + 1e-9))


def nearest_neighbors(word: str, embeddings: dict, top_k: int = 5) -> list:
    """Find top_k closest words using cosine similarity."""
    if word not in embeddings:
        return []
    vec = embeddings[word]
    scores = []
    for other_word, other_vec in embeddings.items():
        if other_word == word or other_word == "<unk>":
            continue
        scores.append((other_word, cosine_similarity(vec, other_vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def analogy_test(word_a: str, word_b: str, word_c: str,
                 embeddings: dict, top_k: int = 3) -> list:
    """
    Solve: word_a - word_b + word_c = ?
    Example: king - man + woman = queen
    """
    if any(w not in embeddings for w in [word_a, word_b, word_c]):
        missing = [w for w in [word_a, word_b, word_c] if w not in embeddings]
        print(f"  Missing from vocab: {missing}")
        return []

    # Compute the analogy vector
    target_vec = (embeddings[word_a]
                  - embeddings[word_b]
                  + embeddings[word_c])

    exclude = {word_a, word_b, word_c, "<unk>"}
    scores = []
    for word, vec in embeddings.items():
        if word in exclude:
            continue
        scores.append((word, cosine_similarity(target_vec, vec)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def similarity_matrix(words: list, embeddings: dict) -> np.ndarray:
    """Build a similarity matrix for a list of words."""
    n = len(words)
    matrix = np.zeros((n, n))
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if w1 in embeddings and w2 in embeddings:
                matrix[i][j] = cosine_similarity(
                    embeddings[w1], embeddings[w2]
                )
    return matrix


def embedding_stats(embeddings: dict) -> dict:
    """Basic statistics about your embeddings."""
    vecs = np.array(list(embeddings.values()))
    return {
        "vocab_size": len(embeddings),
        "embedding_dim": vecs.shape[1],
        "mean_norm": float(np.mean(np.linalg.norm(vecs, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(vecs, axis=1))),
        "mean_value": float(np.mean(vecs)),
        "std_value": float(np.std(vecs)),
    }


def pairwise_similarity_report(embeddings: dict) -> None:
    """Print similarity between all word pairs — shows what model learned."""
    words = [w for w in embeddings if w != "<unk>"]
    print(f"\n{'Word 1':<15} {'Word 2':<15} {'Similarity':>10}")
    print("-" * 42)
    for w1, w2 in combinations(words[:8], 2):  # top 8 words only
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        print(f"{w1:<15} {w2:<15} {sim:>10.4f}")


if __name__ == "__main__":
    print("Loading embeddings...")
    embeddings = load_embeddings("embeddings/embeddings.json")

    # 1. Basic stats
    print("\n--- Embedding Statistics ---")
    stats = embedding_stats(embeddings)
    for k, v in stats.items():
        print(f"  {k:<20} {v}")

    # 2. Nearest neighbors test
    print("\n--- Nearest Neighbors ---")
    test_words = ["dog", "cat", "learning", "language", "fox"]
    for word in test_words:
        if word in embeddings:
            neighbors = nearest_neighbors(word, embeddings, top_k=3)
            neighbor_str = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
            print(f"  {word:<12} → {neighbor_str}")

    # 3. Analogy tests
    print("\n--- Analogy Tests (a - b + c = ?) ---")
    analogies = [
        ("dog", "animal", "cat"),
        ("learning", "machine", "language"),
        ("fox", "animal", "dog"),
    ]
    for a, b, c in analogies:
        print(f"\n  {a} - {b} + {c} = ?")
        results = analogy_test(a, b, c, embeddings)
        if results:
            for word, score in results:
                print(f"    → {word:<15} ({score:.4f})")

    # 4. Similarity matrix
    print("\n--- Similarity Matrix (animal words) ---")
    animal_words = ["dog", "cat", "fox", "animal"]
    available = [w for w in animal_words if w in embeddings]
    if len(available) >= 2:
        matrix = similarity_matrix(available, embeddings)
        print(f"  Words: {available}")
        print("  Matrix:")
        for i, row in enumerate(matrix):
            row_str = "  ".join([f"{v:.3f}" for v in row])
            print(f"    {available[i]:<8} [{row_str}]")

    # 5. Pairwise report
    print("\n--- Pairwise Similarity Report ---")
    pairwise_similarity_report(embeddings)

    print("\n✅ Evaluation complete!")

from collections import defaultdict
import json
import re


class BPETokenizer:
    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.vocab = {}        # token string → id
        self.id_to_token = {}  # id → token string
        self.merges = {}       # (pair) → merged token

    def _get_word_freqs(self, text: str) -> dict:
        """Split text into space-separated words and count frequencies."""
        word_freqs = defaultdict(int)
        for word in text.strip().split():
            # Represent each word as tuple of characters + end-of-word marker
            word = " ".join(list(word)) + " </w>"
            word_freqs[word] += 1
        return word_freqs

    def _get_pair_freqs(self, word_freqs: dict) -> dict:
        """Count frequency of every adjacent pair across all words."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i + 1])] += freq
        return pair_freqs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        """Merge the best pair everywhere it appears."""
        new_word_freqs = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in word_freqs:
            new_word = pattern.sub("".join(pair), word)
            new_word_freqs[new_word] = word_freqs[word]
        return new_word_freqs

    def train(self, text: str):
        """Learn BPE merge rules from text."""
        print(f"Training BPE tokenizer with vocab size {self.vocab_size}...")

        # Start with character-level vocabulary
        word_freqs = self._get_word_freqs(text)

        # Build initial vocab from all unique characters
        alphabet = set()
        for word in word_freqs:
            for char in word.split():
                alphabet.add(char)

        self.vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
        self.vocab["<unk>"] = len(self.vocab)
        self.vocab["<pad>"] = len(self.vocab)

        # Run BPE merges until we hit vocab_size
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break

            # Pick the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged = "".join(best_pair)

            # Save the merge rule
            self.merges[best_pair] = merged

            # Add new token to vocab
            self.vocab[merged] = len(self.vocab)

            # Apply merge to all words
            word_freqs = self._merge_pair(best_pair, word_freqs)

            if (i + 1) % 100 == 0:
                print(
                    f"  Merge {i+1}/{num_merges} — learned: {best_pair} → {merged}")

        # Build reverse vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Training complete. Vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> list:
        """Convert text to list of token ids."""
        tokens = []
        for word in text.strip().split():
            word_chars = list(word) + ["</w>"]
            # Apply merges greedily
            while len(word_chars) > 1:
                pairs = [(word_chars[i], word_chars[i+1])
                         for i in range(len(word_chars)-1)]
                # Find the first merge rule that applies
                mergeable = [(p, self.merges[p])
                             for p in pairs if p in self.merges]
                if not mergeable:
                    break
                # Apply the merge that was learned earliest (lowest vocab id)
                best = min(mergeable, key=lambda x: self.vocab.get(
                    x[1], float("inf")))
                pair, merged = best
                new_chars = []
                i = 0
                while i < len(word_chars):
                    if (i < len(word_chars) - 1 and
                            word_chars[i] == pair[0] and
                            word_chars[i+1] == pair[1]):
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(word_chars[i])
                        i += 1
                word_chars = new_chars

            for ch in word_chars:
                tokens.append(self.vocab.get(ch, self.vocab.get("<unk>")))
        return tokens

    def decode(self, ids: list) -> str:
        """Convert list of token ids back to text."""
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        text = " ".join(tokens)
        # Remove </w> markers and fix spacing
        text = text.replace(" </w>", " ").replace("</w>", " ")
        return text.strip()

    def save(self, path: str):
        """Save vocab and merges to JSON."""
        data = {
            "vocab": self.vocab,
            "merges": [list(k) + [v] for k, v in self.merges.items()]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load vocab and merges from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = {(m[0], m[1]): m[2] for m in data["merges"]}
        print(f"Tokenizer loaded from {path}")


# Quick test — run this file directly to see it work
if __name__ == "__main__":
    sample_text = """
    the quick brown fox jumps over the lazy dog
    the dog barked at the fox and the fox ran away
    natural language processing is fascinating
    """

    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(sample_text)

    test = "the fox ran"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)

    print(f"\nOriginal : {test}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {decoded}")
    print(f"Roundtrip OK: {test == decoded}")

    tokenizer.save("tokenizer/bpe_model.json")

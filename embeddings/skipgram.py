import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import json
import random


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2id = {}
        self.id2word = {}
        self.word_freqs = defaultdict(int)

    def build(self, text: str):
        """Build vocabulary from text."""
        words = text.lower().split()
        for word in words:
            self.word_freqs[word] += 1

        # Only keep words that appear min_freq times
        filtered = {w: f for w, f in self.word_freqs.items()
                    if f >= self.min_freq}

        self.word2id = {w: i for i, w in enumerate(sorted(filtered))}
        self.word2id["<unk>"] = len(self.word2id)
        self.id2word = {i: w for w, i in self.word2id.items()}

        print(f"Vocabulary built: {len(self.word2id)} words")

    def __len__(self):
        return len(self.word2id)

    def get_id(self, word: str) -> int:
        return self.word2id.get(word, self.word2id["<unk>"])


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # Two embedding tables — center word and context word
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Initialize weights small
        nn.init.uniform_(self.center_embeddings.weight, -0.1, 0.1)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center, context, negatives):
        """
        center    : (batch,) center word ids
        context   : (batch,) positive context word ids
        negatives : (batch, num_neg) negative sample ids
        """
        # Get embeddings
        v_center = self.center_embeddings(center)        # (batch, embed_dim)
        v_context = self.context_embeddings(context)     # (batch, embed_dim)
        # (batch, num_neg, embed_dim)
        v_neg = self.context_embeddings(negatives)

        # Positive score — center dot context
        pos_score = torch.sum(v_center * v_context, dim=1)  # (batch,)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-9)

        # Negative score — center dot each negative
        neg_score = torch.bmm(v_neg, v_center.unsqueeze(2)
                              ).squeeze(2)  # (batch, num_neg)
        neg_loss = torch.log(torch.sigmoid(-neg_score) + 1e-9)
        neg_loss = torch.sum(neg_loss, dim=1)

        # Total loss (negative because we maximize)
        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embedding(self, word_id: int) -> np.ndarray:
        """Get the vector for a single word."""
        with torch.no_grad():
            vec = self.center_embeddings(torch.tensor([word_id]))
            return vec.numpy()[0]


class SkipGramDataset:
    def __init__(self, text: str, vocab: Vocabulary,
                 window_size: int = 2, num_negatives: int = 5):
        self.vocab = vocab
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.pairs = []

        # Build (center, context) pairs
        words = text.lower().split()
        word_ids = [vocab.get_id(w) for w in words]

        for i, center in enumerate(word_ids):
            # Dynamic window size
            win = random.randint(1, window_size)
            for j in range(max(0, i - win), min(len(word_ids), i + win + 1)):
                if i != j:
                    self.pairs.append((center, word_ids[j]))

        # Build noise distribution P(w)^(3/4) for negative sampling
        freqs = np.zeros(len(vocab))
        for word, idx in vocab.word2id.items():
            freqs[idx] = vocab.word_freqs.get(word, 0)
        freqs = freqs ** 0.75
        self.noise_dist = freqs / freqs.sum()

        print(f"Dataset built: {len(self.pairs)} training pairs")

    def get_batch(self, batch_size: int):
        """Get a random batch of (center, context, negatives)."""
        batch = random.sample(self.pairs, min(batch_size, len(self.pairs)))
        centers = torch.tensor([p[0] for p in batch])
        contexts = torch.tensor([p[1] for p in batch])

        # Sample negatives using noise distribution
        negatives = np.random.choice(
            len(self.vocab),
            size=(len(batch), self.num_negatives),
            p=self.noise_dist
        )
        negatives = torch.tensor(negatives, dtype=torch.long)

        return centers, contexts, negatives


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / (norm + 1e-9)


def find_nearest_neighbors(word: str, vocab: Vocabulary,
                           model: SkipGramModel, top_k: int = 5):
    """Find top_k most similar words to the given word."""
    if word not in vocab.word2id:
        print(f"'{word}' not in vocabulary")
        return []

    word_id = vocab.get_id(word)
    word_vec = model.get_embedding(word_id)

    similarities = []
    for other_word, other_id in vocab.word2id.items():
        if other_word == word or other_word == "<unk>":
            continue
        other_vec = model.get_embedding(other_id)
        sim = cosine_similarity(word_vec, other_vec)
        similarities.append((other_word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def save_embeddings(model: SkipGramModel, vocab: Vocabulary, path: str):
    """Save embeddings to JSON."""
    embeddings = {}
    for word, idx in vocab.word2id.items():
        embeddings[word] = model.get_embedding(idx).tolist()
    with open(path, "w") as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved to {path}")


# Quick test
if __name__ == "__main__":
    TEXT = """
    the quick brown fox jumps over the lazy dog
    the dog and the fox are both animals
    the cat sat on the mat and looked at the dog
    machine learning and deep learning are related fields
    natural language processing uses machine learning
    the dog chased the cat and the cat ran away
    language models learn from large amounts of text
    the fox and the dog played in the field
    """

    EMBED_DIM = 50
    BATCH_SIZE = 32
    EPOCHS = 500

    # Build vocab
    vocab = Vocabulary(min_freq=2)
    vocab.build(TEXT)

    # Build dataset
    dataset = SkipGramDataset(TEXT, vocab, window_size=2, num_negatives=5)

    # Build model
    model = SkipGramModel(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("\nTraining SkipGram model...")
    for epoch in range(EPOCHS):
        centers, contexts, negatives = dataset.get_batch(BATCH_SIZE)
        optimizer.zero_grad()
        loss = model(centers, contexts, negatives)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

    # Test nearest neighbors
    print("\n--- Nearest Neighbors ---")
    for test_word in ["dog", "cat", "learning", "language"]:
        neighbors = find_nearest_neighbors(test_word, vocab, model)
        if neighbors:
            print(f"\n'{test_word}' is closest to:")
            for word, score in neighbors:
                print(f"   {word:15s} {score:.4f}")

    # Save embeddings
    save_embeddings(model, vocab, "embeddings/embeddings.json")

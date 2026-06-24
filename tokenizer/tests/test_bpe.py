from tokenizer.bpe import BPETokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Sample training text
TRAIN_TEXT = """
the quick brown fox jumps over the lazy dog
the dog barked at the fox and the fox ran away
natural language processing is fascinating
machine learning is powerful and exciting
"""


def get_trained_tokenizer():
    t = BPETokenizer(vocab_size=100)
    t.train(TRAIN_TEXT)
    return t


def test_vocab_size():
    """Vocab should not exceed requested size."""
    t = get_trained_tokenizer()
    assert len(t.vocab) <= 100, f"Vocab too large: {len(t.vocab)}"
    print("✅ test_vocab_size passed")


def test_encode_returns_list_of_ints():
    """Encode must return a list of integers."""
    t = get_trained_tokenizer()
    ids = t.encode("the fox")
    assert isinstance(ids, list), "encode() must return a list"
    assert all(isinstance(i, int) for i in ids), "All ids must be integers"
    print("✅ test_encode_returns_list_of_ints passed")


def test_roundtrip():
    """decode(encode(text)) must equal original text."""
    t = get_trained_tokenizer()
    test_cases = [
        "the fox ran",
        "natural language",
        "the dog",
    ]
    for text in test_cases:
        encoded = t.encode(text)
        decoded = t.decode(encoded)
        assert decoded == text, f"Roundtrip failed: '{text}' → '{decoded}'"
    print("✅ test_roundtrip passed")


def test_unknown_token():
    """Words not in vocab should not crash encode()."""
    t = get_trained_tokenizer()
    try:
        ids = t.encode("zzzzunknownword")
        assert isinstance(ids, list)
        print("✅ test_unknown_token passed")
    except Exception as e:
        assert False, f"encode() crashed on unknown word: {e}"


def test_save_and_load(tmp_path="tokenizer/test_model.json"):
    """Saved and reloaded tokenizer must produce same output."""
    t = get_trained_tokenizer()
    t.save(tmp_path)

    t2 = BPETokenizer(vocab_size=100)
    t2.load(tmp_path)

    ids1 = t.encode("the fox")
    ids2 = t2.encode("the fox")
    assert ids1 == ids2, "Loaded tokenizer produces different output"
    print("✅ test_save_and_load passed")
    os.remove(tmp_path)


if __name__ == "__main__":
    print("\n--- Running BPE Tokenizer Tests ---\n")
    test_vocab_size()
    test_encode_returns_list_of_ints()
    test_roundtrip()
    test_unknown_token()
    test_save_and_load()
    print("\n🎉 All tests passed!")

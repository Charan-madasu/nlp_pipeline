# End-to-End NLP Pipeline
> BPE Tokenization | SkipGram Embeddings | GPT-2 Fine-Tuning

A complete NLP pipeline built **from scratch** — no shortcuts, no black boxes.
Every component implemented manually in Python and PyTorch.

---

## What This Project Does

Most people use libraries like HuggingFace and call it done.
This project implements everything from the ground up:

- **BPE Tokenizer** — same algorithm used inside GPT-4, built manually
- **SkipGram Word2Vec** — word embeddings that learn meaning from raw text
- **GPT-2 Fine-Tuning** — instruction-following model trained on Alpaca dataset
- **4 Decoding Methods** — greedy, temperature, top-k, top-p all from scratch
- **Evaluation Suite** — perplexity, repetition rate, diversity metrics

---

## Project Structure

---

## BPE Tokenizer — Results

Built from scratch. Trains merge rules, encodes text to integers,
decodes back to text losslessly.

**5/5 unit tests passing:**

| Test | Result |
|---|---|
| Vocab size constraint | ✅ Passed |
| Encode returns integers | ✅ Passed |
| Roundtrip lossless decode | ✅ Passed |
| Unknown token handling | ✅ Passed |
| Save and load from JSON | ✅ Passed |

```python
tokenizer = BPETokenizer(vocab_size=100)
tokenizer.train(text)
ids = tokenizer.encode("the fox ran")   # → [31, 35, 74]
text = tokenizer.decode(ids)            # → "the fox ran"
```

---

## SkipGram Embeddings — Results

Trained on raw text with zero human labels.
Model learned semantic relationships purely from reading context.

| Word | Top 3 Closest Words |
|---|---|
| dog | learning(0.76), cat(0.70), language(0.60) |
| cat | fox(0.77), dog(0.70), language(0.67) |
| learning | dog(0.76), language(0.63), fox(0.61) |
| fox | cat(0.77), language(0.66), learning(0.61) |

**Key implementation details:**
- Negative sampling with noise distribution P(w)^(3/4)
- Dynamic context window size
- Manual cosine similarity search
- No Gensim or sklearn used

---

## GPT-2 Fine-Tuning — Results

| Setting | Value |
|---|---|
| Base Model | GPT-2 (124M parameters) |
| Dataset | Stanford Alpaca |
| Training Examples | 5,000 |
| Epochs | 3 |
| Final Train Loss | 0.284 |
| Final Val Loss | 0.3296 |
| Final Perplexity | 1.39 |
| Mixed Precision | ✅ torch.cuda.amp |
| LR Scheduler | Cosine warmup (manual) |
| Gradient Clipping | ✅ norm = 1.0 |

---

## Decoding Methods — Sample Outputs

**Prompt:** "Explain what machine learning is"

| Method | Output |
|---|---|
| Greedy | Machine learning is a type of artificial intelligence that uses data to learn... |
| Temperature (0.7) | Machine learning is an approach to understanding how machines learn from data... |
| Top-K (k=50) | Machine learning is an AI-driven approach that seeks to collect data... |
| Top-P (p=0.9) | Machine learning is a kind of artificial intelligence which uses data... |

All decoding methods implemented manually — not using `model.generate()` as a black box.

---

## How To Run

### 1. Clone and Setup
```bash
git clone https://github.com/Charan-madasu/nlp_pipeline.git
cd nlp_pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run BPE Tokenizer
```bash
python tokenizer/bpe.py
```

### 3. Run Unit Tests
```bash
python -m pytest tokenizer/tests/ -v
```

### 4. Run SkipGram Training
```bash
python embeddings/skipgram.py
```

### 5. Run Evaluation Suite
```bash
cd eval
python run_eval.py
```

---

## Technical Highlights

| What | How |
|---|---|
| Tokenizer | BPE from scratch — no tokenizers library |
| Embeddings | SkipGram from scratch — no Gensim |
| Training loop | Manual — no HuggingFace Trainer |
| Decoding | Manual — no model.generate() black box |
| LR Scheduler | Cosine warmup — manually implemented |
| Mixed precision | torch.cuda.amp — memory efficient |
| Testing | pytest — 5 unit tests, all passing |
| Version control | Git + GitHub with clean commit history |

---

## Skills

`Python` `PyTorch` `HuggingFace Transformers` `NLP` `Deep Learning`
`BPE Tokenization` `Word Embeddings` `GPT-2` `Text Generation`
`Model Evaluation` `Unit Testing` `Git` `Google Colab`

---

## Author
**Charan Krishna Madasu**
[GitHub](https://github.com/Charan-madasu)
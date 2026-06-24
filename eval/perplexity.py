import torch
import math


def calculate_perplexity(model, tokenizer, texts: list, device="cpu") -> float:
    """Calculate average perplexity across a list of texts."""
    model.eval()
    total_loss = 0
    count = 0

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1

    avg_loss = total_loss / count
    return math.exp(avg_loss)

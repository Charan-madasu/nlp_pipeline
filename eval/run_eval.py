from diversity import diversity_report
from repetition import avg_repetition
from perplexity import calculate_perplexity
from perplexity import calculate_perplexity
from repetition import avg_repetition
from diversity import diversity_report
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Test prompts
PROMPTS = [
    "### Instruction:\nExplain what machine learning is.\n\n### Response:\n",
    "### Instruction:\nWhat are three benefits of exercise?\n\n### Response:\n",
    "### Instruction:\nHow does the internet work?\n\n### Response:\n",
    "### Instruction:\nWhat is photosynthesis?\n\n### Response:\n",
    "### Instruction:\nExplain gravity in simple terms.\n\n### Response:\n",
]


def generate_outputs(model, tokenizer, prompts, decode_fn, device="cpu"):
    """Generate outputs for all prompts using a given decoding function."""
    outputs = []
    for prompt in prompts:
        result = decode_fn(model, tokenizer, prompt)
        outputs.append(result)
    return outputs


def run_full_eval(model, tokenizer, decode_fns: dict, device="cpu"):
    """
    Run all metrics for each decoding strategy.
    decode_fns = {"Greedy": greedy_fn, "Top-P": topp_fn, ...}
    """
    print("\n" + "="*70)
    print(f"{'Method':<15} {'Perplexity':>12} {'Repetition':>12} {'Distinct-1':>12} {'Distinct-2':>12}")
    print("="*70)

    results = {}
    for method_name, decode_fn in decode_fns.items():
        # Generate outputs
        outputs = generate_outputs(
            model, tokenizer, PROMPTS, decode_fn, device)

        # Calculate metrics
        ppl = calculate_perplexity(model, tokenizer, outputs, device)
        rep = avg_repetition(outputs, n=2)
        div = diversity_report(outputs)

        results[method_name] = {
            "perplexity": ppl,
            "repetition": rep,
            "distinct_1": div["distinct_1"],
            "distinct_2": div["distinct_2"],
            "outputs": outputs
        }

        print(
            f"{method_name:<15} {ppl:>12.4f} {rep:>12.4f} {div['distinct_1']:>12.4f} {div['distinct_2']:>12.4f}")

    print("="*70)

    # Print sample outputs
    print("\n--- Sample Outputs (Prompt 1: What is machine learning?) ---")
    for method_name, data in results.items():
        print(f"\n{method_name}:")
        print(f"  {data['outputs'][0][:150]}...")

    return results


if __name__ == "__main__":
    print("Loading model for evaluation...")
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch.nn.functional as F

    # Load model — CPU since we're on local machine
    MODEL_PATH = "gpt2"

    try:
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        print("✅ Fine-tuned model loaded!")
    except:
        print("⚠️  Fine-tuned model not found, using base GPT-2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Define decoding functions inline
    def greedy(model, tokenizer, prompt, max_new_tokens=80):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    def temperature(model, tokenizer, prompt, max_new_tokens=80):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    def topk(model, tokenizer, prompt, max_new_tokens=80):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50
            )
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    def topp(model, tokenizer, prompt, max_new_tokens=80):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9
            )
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    # Run evaluation
    decode_fns = {
        "Greedy":      greedy,
        "Temperature": temperature,
        "Top-K":       topk,
        "Top-P":       topp,
    }

    results = run_full_eval(model, tokenizer, decode_fns, device="cpu")
    print("\n✅ Evaluation complete!")

from transformers import AutoModelForCausalLM, AutoTokenizer

from logitscope import LogitScope

# Load Model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Initialze tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Metric to show
metrics = ["surprisal", "entropy", "varentropy", "skewentropy", "perplexity"]
# Show top 5 highest probability tokens
show_top_tokens = True

# Input text to calculate metrics
conv = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]
text = tokenizer.apply_chat_template(conv, tokenize=False)

scope = LogitScope(tokenizer, model, device="cpu", seed=None)
result = scope.measure(text)

print(f"Input text: \n{text}\n")

# Tokens are automatically decoded with actual spaces, newlines, tabs
# Use repr() to show escape sequences in CLI output for readability
for i, t in enumerate(result.iter_tokens(metrics=metrics)):
    print(f"Token: {repr(t['token'])}")
    for m in metrics:
        if m in t:
            print("> " + m + ":\t" + str(t[m]))
    print()

    if show_top_tokens and i > 0:
        # top_k also returns properly decoded tokens
        top_prob = result.top_k(i, k=5)
        print("  Top-5 alternatives:")
        for token, prob in top_prob:
            print(f"    {repr(token):22s} {prob:.4f}")
        print()

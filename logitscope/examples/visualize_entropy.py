import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from logitscope import LogitScope

# Load Model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = (
    "We hold these truths to be self-evident, that all men are created equal, that "
    "they are endowed by their Creator with certain unalienable Rights, that among "
    "these are Life, Liberty and the pursuit of Happiness.--That to secure these "
    "rights, Governments are instituted among Men, deriving their just powers from "
    "the consent of the governed, --That whenever any Form of Government becomes "
    "destructive of these ends, it is the Right of the People to alter or to abolish "
    "it, and to institute new Government, laying its foundation on such principles "
    "and organizing its powers in such form, as to them shall seem most likely to "
    "effect their Safety and Happiness. Prudence, indeed, will dictate that "
    "Governments long established should not be changed for light and transient "
    "causes; and accordingly all experience hath shewn, that mankind are more "
    "disposed to suffer, while evils are sufferable, than to right themselves by "
    "abolishing the forms to which they are accustomed. But when a long train of "
    "abuses and usurpations, pursuing invariably the same Object evinces a design "
    "to reduce them under absolute Despotism, it is their right, it is their duty, "
    "to throw off such Government, and to provide new Guards for their future "
    "security.--Such has been the patient sufferance of these Colonies; and such is "
    "now the necessity which constrains them to alter their former Systems of "
    "Government. The history of the present King of Great Britain is a history of "
    "repeated injuries and usurpations, all having in direct object the "
    "establishment of an absolute Tyranny over these States. To prove this, let "
    "Facts be submitted to a candid world."
)

scope = LogitScope(tokenizer, model, device="cpu", seed=None)
result = scope.measure(text)

print(f"Input text: \n{text}\n")

# Collect entropy and varentropy values
entropy_values = []
varentropy_values = []
tokens = []

for t in result.iter_tokens(metrics=["entropy", "varentropy"]):
    if t["index"] > 0:
        entropy_values.append(t["entropy"])
        varentropy_values.append(t["varentropy"])
        tokens.append(t["token"])

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(entropy_values, varentropy_values, alpha=0.6, s=100)

plt.xlabel("Entropy (uncertainty)", fontsize=10)
plt.ylabel("Varentropy (distribution spread)", fontsize=10)
plt.title("Entropy vs Varentropy", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)

plt.show()

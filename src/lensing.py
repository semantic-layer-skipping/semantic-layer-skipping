import transformer_lens

# Load a model (eg GPT-2 Small)
#model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
model = transformer_lens.HookedTransformer.from_pretrained("google-bert/bert-base-uncased", fold_ln=False, device="mps")

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")

print(logits)
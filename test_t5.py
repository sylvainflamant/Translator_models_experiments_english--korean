from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "nlp-with-deeplearning/enko-t5-small-v0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Increase the temperature in the living room to 24 degrees."

# Explicitly set attention_mask and use pad_token_id != decoder_start_token_id
encoded = tokenizer(text, return_tensors="pt", padding=True, return_attention_mask=True)
print(f"input_ids: {encoded['input_ids']}")
print(f"attention_mask: {encoded['attention_mask']}")

# Try with sampling instead of greedy
print("\nSampling (temperature=0.7):")
outputs = model.generate(
    **encoded,
    max_length=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Result: '{result}'")

# Try with repetition penalty
print("\nBeam=5 + repetition_penalty=2.0:")
outputs = model.generate(
    **encoded,
    max_length=128,
    num_beams=5,
    repetition_penalty=2.0,
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Result: '{result}'")

# Try simple sentences
print("\nSimple sentences with beam=5:")
for t in ["Hello", "Thank you", "I love you", "Good morning"]:
    enc = tokenizer(t, return_tensors="pt", return_attention_mask=True)
    out = model.generate(**enc, max_length=128, num_beams=5)
    r = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  '{t}' -> '{r}'")

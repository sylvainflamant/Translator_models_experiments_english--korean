from transformers import MarianMTModel, MarianTokenizer
import transformers

print(f"transformers version: {transformers.__version__}")

# The model card says it was converted with transformers 4.16.2
# Let's try loading with from_pretrained using revision
model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Check if there's a pytorch_model.bin vs safetensors issue
# Try forcing pytorch format
try:
    model = MarianMTModel.from_pretrained(model_name, use_safetensors=False)
    text = "Increase the temperature in the living room to 24 degrees."
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"pytorch_model.bin: '{result}'")
except Exception as e:
    print(f"pytorch_model.bin error: {e}")

try:
    model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)
    text = "Increase the temperature in the living room to 24 degrees."
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"safetensors: '{result}'")
except Exception as e:
    print(f"safetensors error: {e}")

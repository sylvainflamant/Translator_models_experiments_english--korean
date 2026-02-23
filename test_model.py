from transformers import MarianTokenizer, MarianMTModel

model_name = 'Helsinki-NLP/opus-mt-tc-big-en-ko'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = 'Increase the temperature in the living room to 24 degrees.'
# Add language token for Korean
text_with_token = '>>kor<< ' + text
inputs = tokenizer(text_with_token, return_tensors='pt', padding=True)
translated = model.generate(**inputs)
result = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f'Input: {text}')
print(f'Output: {result}')


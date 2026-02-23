from transformers import pipeline

pipe = pipeline("translation_en_to_ko", model="Helsinki-NLP/opus-mt-tc-big-en-ko")
result = pipe("Increase the temperature in the living room to 24 degrees.")
print(result)

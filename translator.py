import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import torch
import time
import sys

# Language codes for NLLB-200
ENG_CODE = "eng_Latn"  # English
KOR_CODE = "kor_Hang"  # Korean

# Original model (NLLB)
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text, src_lang, tgt_lang):
    """Translate text from source language to target language."""
    inputs = tokenizer(text, return_tensors="pt")
    start_time = time.time()
    translated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=512
    )
    elapsed_time = time.time() - start_time
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True), elapsed_time

def english_to_korean(text):
    """Translate English to Korean."""
    result, elapsed = translate(text, ENG_CODE, KOR_CODE)
    return result, elapsed

def korean_to_english(text):
    """Translate Korean to English."""
    result, elapsed = translate(text, KOR_CODE, ENG_CODE)
    return result, elapsed

def round_trip_translate(text):
    """
    Perform round-trip translation: English -> Korean -> English -> Korean -> English
    Returns the final English text, translation path, and timing information.
    """
    start_time = time.time()
    
    # English -> Korean
    korean1, t1 = english_to_korean(text)
    # Korean -> English
    english1, t2 = korean_to_english(korean1)
    # English -> Korean
    korean2, t3 = english_to_korean(english1)
    # Korean -> English
    english2, t4 = korean_to_english(korean2)
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    return {
        'original': text,
        'korean1': korean1,
        'english1': english1,
        'korean2': korean2,
        'final': english2,
        'elapsed_time': total_elapsed,
        'step_times': [t1, t2, t3, t4]
    }

# M2M100 model for English-Korean translation (loaded on demand)
model_name_2 = "facebook/m2m100_418M"
tokenizer_2 = None
model_2 = None

def load_model_2():
    """Load M2M100 model on demand."""
    global tokenizer_2, model_2
    if tokenizer_2 is None:
        print("Loading M2M100 model...")
        tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
        model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_name_2)

def translate_2(text, src_lang, tgt_lang):
    """Translate text using M2M100 model."""
    load_model_2()
    # Map NLLB codes to M2M100 codes
    lang_map = {"eng_Latn": "en", "kor_Hang": "ko"}
    src = lang_map[src_lang]
    tgt = lang_map[tgt_lang]
    
    tokenizer_2.src_lang = src
    inputs = tokenizer_2(text, return_tensors="pt")
    start_time = time.time()
    translated_tokens = model_2.generate(
        **inputs,
        forced_bos_token_id=tokenizer_2.get_lang_id(tgt),
        max_length=512
    )
    elapsed_time = time.time() - start_time
    return tokenizer_2.decode(translated_tokens[0], skip_special_tokens=True), elapsed_time

def english_to_korean_2(text):
    """Translate English to Korean using Helsinki-NLP model."""
    result, elapsed = translate_2(text, ENG_CODE, KOR_CODE)
    return result, elapsed

def korean_to_english_2(text):
    """Translate Korean to English using Helsinki-NLP model."""
    result, elapsed = translate_2(text, KOR_CODE, ENG_CODE)
    return result, elapsed

def round_trip_translate_2(text):
    """
    Perform round-trip translation using Helsinki-NLP models: English -> Korean -> English -> Korean -> English
    """
    start_time = time.time()
    
    korean1, t1 = english_to_korean_2(text)
    english1, t2 = korean_to_english_2(korean1)
    korean2, t3 = english_to_korean_2(english1)
    english2, t4 = korean_to_english_2(korean2)
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    return {
        'original': text,
        'korean1': korean1,
        'english1': english1,
        'korean2': korean2,
        'final': english2,
        'elapsed_time': total_elapsed,
        'step_times': [t1, t2, t3, t4]
    }

def compute_semantic_similarity(text1, text2):
    """
    Compute semantic similarity between two texts using cosine similarity.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()
    except ImportError:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        if not union:
            return 1.0 if text1 == text2 else 0.0
        return len(intersection) / len(union)

# Helsinki-NLP models for English-Korean translation (loaded on demand)
# Using NLLB-200-distilled-1.3B as a higher quality alternative
model_name_3 = "facebook/nllb-200-distilled-1.3B"
tokenizer_3 = None
model_3 = None

def load_model_3():
    """Load NLLB 1.3B model on demand."""
    global tokenizer_3, model_3
    if tokenizer_3 is None:
        print("Loading NLLB 1.3B model...")
        tokenizer_3 = AutoTokenizer.from_pretrained(model_name_3)
        model_3 = AutoModelForSeq2SeqLM.from_pretrained(model_name_3)

def translate_3(text, src_lang, tgt_lang):
    """Translate text using NLLB 1.3B model."""
    load_model_3()
    inputs = tokenizer_3(text, return_tensors="pt")
    start_time = time.time()
    translated_tokens = model_3.generate(
        **inputs,
        forced_bos_token_id=tokenizer_3.convert_tokens_to_ids(tgt_lang),
        max_length=512
    )
    elapsed_time = time.time() - start_time
    return tokenizer_3.decode(translated_tokens[0], skip_special_tokens=True), elapsed_time

def english_to_korean_3(text):
    """Translate English to Korean using NLLB 1.3B model."""
    result, elapsed = translate_3(text, ENG_CODE, KOR_CODE)
    return result, elapsed

def korean_to_english_3(text):
    """Translate Korean to English using NLLB 1.3B model."""
    result, elapsed = translate_3(text, KOR_CODE, ENG_CODE)
    return result, elapsed

def round_trip_translate_3(text):
    """Perform round-trip translation using NLLB 1.3B model."""
    start_time = time.time()
    korean1, t1 = english_to_korean_3(text)
    english1, t2 = korean_to_english_3(korean1)
    korean2, t3 = english_to_korean_3(english1)
    english2, t4 = korean_to_english_3(korean2)
    end_time = time.time()
    return {
        'original': text,
        'korean1': korean1,
        'english1': english1,
        'korean2': korean2,
        'final': english2,
        'elapsed_time': end_time - start_time,
        'step_times': [t1, t2, t3, t4]
    }

def main():
    """Main function to demonstrate round-trip translation and similarity scoring."""
    model_choice = "1"  # Default to original model
    test_text = "Increase the temperature in the living room to 24 degrees."
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        model_choice = sys.argv[1]
        if len(sys.argv) > 2:
            test_text = sys.argv[2]
    
    print(f"Translating: {test_text}")
    print(f"Language pair: English <-> Korean")
    print("-" * 50)
    
    if model_choice == "1":
        # Original NLLB model
        print("\n=== Original Model (facebook/nllb-200-distilled-600M) ===")
        result = round_trip_translate(test_text)
        similarity = compute_semantic_similarity(result['original'], result['final'])
        print(f"Original English: {result['original']}")
        print(f"→ Korean 1: {result['korean1']}")
        print(f"→ English 1: {result['english1']}")
        print(f"→ Korean 2: {result['korean2']}")
        print(f"→ Final English: {result['final']}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Model running time: {sum(result['step_times']):.2f} seconds")
    elif model_choice == "2":
        # M2M100 model
        print("\n=== M2M100 Model (facebook/m2m100_418M) ===")
        result_2 = round_trip_translate_2(test_text)
        similarity_2 = compute_semantic_similarity(result_2['original'], result_2['final'])
        print(f"Original English: {result_2['original']}")
        print(f"→ Korean 1: {result_2['korean1']}")
        print(f"→ English 1: {result_2['english1']}")
        print(f"→ Korean 2: {result_2['korean2']}")
        print(f"→ Final English: {result_2['final']}")
        print(f"Similarity: {similarity_2:.4f}")
        print(f"Model running time: {sum(result_2['step_times']):.2f} seconds")
    elif model_choice == "3":
        # Helsinki-NLP models
        print("\n=== Helsinki-NLP Models (opus-mt-tc-big-en-ko / opus-mt-ko-en) ===")
        result_3 = round_trip_translate_3(test_text)
        similarity_3 = compute_semantic_similarity(result_3['original'], result_3['final'])
        print(f"Original English: {result_3['original']}")
        print(f"→ Korean 1: {result_3['korean1']}")
        print(f"→ English 1: {result_3['english1']}")
        print(f"→ Korean 2: {result_3['korean2']}")
        print(f"→ Final English: {result_3['final']}")
        print(f"Similarity: {similarity_3:.4f}")
        print(f"Model running time: {sum(result_3['step_times']):.2f} seconds")
    else:
        print(f"Invalid model choice: {model_choice}")
        print("Usage: python translator.py <model> [text]")
        print("  <model>: 1 for NLLB, 2 for M2M100, 3 for Helsinki-NLP")
        print("  [text]: optional text to translate (default provided if omitted)")

if __name__ == "__main__":
    main()

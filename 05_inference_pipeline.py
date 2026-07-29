"""
STEP 5: Inference with HuggingFace Pipeline
=============================================

THE BIG PICTURE:
    Steps 1-4: you understood HOW transformers work inside.
    Step 5:    you learn to USE them for real tasks, zero training.

    HuggingFace `pipeline` wraps the entire flow:
      Text → Tokenize → Model → Post-process → Human-readable output

    You don't write tokenizer code, model code, or output parsing.
    One line of code = a working NLP system.

WHY THIS MATTERS:
    ~40% of real-world NLP tasks DON'T need fine-tuning.
    Pretrained models already handle:
      - Sentiment analysis
      - Named Entity Recognition
      - Zero-shot classification (classify WITHOUT training data!)
      - Fill-mask (predict missing words — how BERT was trained)
      - Text generation (GPT-style completion)

    Knowing when to use pipeline vs when to fine-tune
    is a critical skill that separates juniors from seniors.

WHAT YOU'LL LEARN:
    1. Sentiment Analysis — is text positive or negative?
    2. Named Entity Recognition — find names, places, organizations
    3. Fill-Mask — predict missing words (BERT's native task)
    4. Zero-Shot Classification — classify without ANY training
    5. Text Generation — GPT-style text completion
    6. Domain-specific models — why default models fail
    7. Feature extraction — get embeddings for downstream use
    8. When to use pipeline vs when to fine-tune
"""

from transformers import pipeline
import time

# ============================================================
# PART 1: Sentiment Analysis — The "Hello World" of NLP
# ============================================================
# WHAT IT DOES:
#   Input:  a sentence
#   Output: POSITIVE or NEGATIVE + confidence score
#
# WHAT'S HAPPENING INSIDE:
#   1. Tokenizer converts text to IDs
#   2. BERT (or similar) produces contextual embeddings
#   3. [CLS] token's embedding goes through a classification head
#   4. Softmax gives probability for each label
#
#   The model used by default: distilbert-base-uncased-finetuned-sst-2-english
#   It's a DistilBERT (smaller BERT) already fine-tuned on movie reviews.

print("=" * 60)
print("PART 1: Sentiment Analysis")
print("=" * 60)

start = time.time()
sentiment = pipeline("sentiment-analysis")
print(f"Model loaded in {time.time() - start:.1f}s")

texts = [
    "I absolutely love this product, it changed my life!",
    "This is the worst experience I've ever had.",
    "The movie was okay, nothing special.",
    "The stock market crashed today, investors are panicking.",
    "Revenue grew 25% year over year, exceeding all expectations.",
]

print(f"\nSentiment Analysis Results:")
for text in texts:
    result = sentiment(text)[0]
    label = result["label"]
    score = result["score"]
    bar = "█" * int(score * 30)
    print(f"\n  Text:  '{text}'")
    print(f"  Label: {label:>8}  Confidence: {score:.4f}  {bar}")

# HOW TO VERIFY:
#   - Clearly positive text → POSITIVE with high confidence (>0.95)
#   - Clearly negative text → NEGATIVE with high confidence (>0.95)
#   - Ambiguous text → lower confidence (0.5-0.8)
#   - Financial text may be wrong — this model was trained on MOVIE reviews,
#     not financial data. That's when you need fine-tuning (Step 6).

# Batch processing — faster than one by one
batch_results = sentiment(texts)
print(f"\n✓ Batch processing works: {len(batch_results)} results at once")


# ============================================================
# PART 2: Named Entity Recognition (NER) — Find Real-World Entities
# ============================================================
# WHAT IT DOES:
#   Input:  a sentence
#   Output: entities found (person names, organizations, locations, etc.)
#
# WHAT'S HAPPENING INSIDE:
#   Unlike sentiment (one label per sentence), NER gives
#   a label PER TOKEN. Each token is classified as:
#     B-PER (beginning of person name)
#     I-PER (inside/continuation of person name)
#     B-ORG (beginning of organization)
#     B-LOC (beginning of location)
#     O     (not an entity)
#
# WHY THIS IS USEFUL:
#   Extract structured data from unstructured text.
#   "Mohit joined Google in Bangalore" →
#     Person: Mohit, Organization: Google, Location: Bangalore

print("\n" + "=" * 60)
print("PART 2: Named Entity Recognition (NER)")
print("=" * 60)

ner = pipeline("ner", aggregation_strategy="simple")

texts = [
    "Elon Musk founded SpaceX in Hawthorne, California.",
    "Apple released the iPhone 15 at their Cupertino headquarters.",
    "Mohit works at a tech company in Bangalore, India.",
    "The Reserve Bank of India announced new monetary policy in Mumbai.",
]

for text in texts:
    entities = ner(text)
    print(f"\n  Text: '{text}'")
    if entities:
        for ent in entities:
            print(f"    → {ent['entity_group']:>12}: '{ent['word']}'  "
                  f"(confidence: {ent['score']:.4f})")
    else:
        print(f"    → No entities found")

# HOW TO VERIFY:
#   - Person names → PER
#   - Company names → ORG
#   - Cities, countries → LOC
#   - Confidence > 0.9 for clear entities
#   - May miss domain-specific entities (financial tickers, medical terms)
#     → that's when you fine-tune


# ============================================================
# PART 3: Fill-Mask — BERT's Native Superpower
# ============================================================
# WHAT IT DOES:
#   Input:  a sentence with [MASK] token replacing a word
#   Output: top predictions for what the masked word should be
#
# WHY THIS MATTERS:
#   This is exactly HOW BERT was pretrained!
#   During pretraining, BERT saw millions of sentences with
#   random words masked out and learned to predict them.
#   This is called Masked Language Modeling (MLM).
#
#   The reason BERT understands language is because predicting
#   masked words FORCES it to understand context, grammar,
#   and world knowledge.
#
# PRACTICAL USES:
#   - Autocomplete / suggestions
#   - Data augmentation (replace words with predictions)
#   - Probing what the model "knows"

print("\n" + "=" * 60)
print("PART 3: Fill-Mask (BERT's Native Task)")
print("=" * 60)

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

masked_sentences = [
    "The capital of France is [MASK].",
    "I went to the [MASK] to buy groceries.",
    "She is a brilliant [MASK] at the university.",
    "The [MASK] barked loudly at the mailman.",
    "Python is a popular [MASK] language.",
]

for sentence in masked_sentences:
    results = fill_mask(sentence)
    print(f"\n  '{sentence}'")
    print(f"  Top 5 predictions:")
    for r in results[:5]:
        score = r["score"]
        bar = "█" * int(score * 40)
        print(f"    {r['token_str']:>15}  ({score:.4f})  {bar}")

# Context changes predictions — prove that BERT understands context
print(f"\n  Context changes predictions:")
context_pairs = [
    ("The [MASK] chased the mouse.", "The [MASK] chased the criminal."),
    ("She plays the [MASK] beautifully.", "She plays [MASK] every weekend."),
]

for s1, s2 in context_pairs:
    r1 = fill_mask(s1)[0]
    r2 = fill_mask(s2)[0]
    print(f"\n    '{s1}' → '{r1['token_str']}' ({r1['score']:.3f})")
    print(f"    '{s2}' → '{r2['token_str']}' ({r2['score']:.3f})")
    print(f"    → Same [MASK] position, different context = different prediction!")

# HOW TO VERIFY:
#   - Predictions should be grammatically correct
#   - Predictions should make semantic sense
#   - Different contexts → different predictions for same position
#   - Top prediction usually has higher confidence than others


# ============================================================
# PART 4: Zero-Shot Classification — No Training Data Needed
# ============================================================
# WHAT IT DOES:
#   Classify text into categories you define AT RUNTIME.
#   No training, no labeled data, no fine-tuning.
#
# WHAT'S HAPPENING INSIDE:
#   Uses a model trained on Natural Language Inference (NLI).
#   For each label, it asks: "Does this text entail this label?"
#
#   Text: "The stock price surged after earnings report"
#   → "This text is about finance"      → entailment score: 0.95
#   → "This text is about sports"       → entailment score: 0.02
#   → "This text is about technology"   → entailment score: 0.03
#
# WHY THIS IS POWERFUL:
#   Traditional classification: collect data → label it → train → deploy
#   Zero-shot: just define your categories → deploy
#   Perfect for prototyping or when you have no labeled data.

print("\n" + "=" * 60)
print("PART 4: Zero-Shot Classification")
print("=" * 60)

zero_shot = pipeline("zero-shot-classification")

texts_and_labels = [
    {
        "text": "The Federal Reserve raised interest rates by 25 basis points",
        "labels": ["finance", "sports", "technology", "politics"],
    },
    {
        "text": "The new iPhone features a revolutionary AI chip",
        "labels": ["finance", "sports", "technology", "politics"],
    },
    {
        "text": "Manchester United won the Champions League final",
        "labels": ["finance", "sports", "technology", "politics"],
    },
    {
        "text": "The patient was diagnosed with type 2 diabetes",
        "labels": ["health", "finance", "sports", "education"],
    },
]

for item in texts_and_labels:
    result = zero_shot(item["text"], candidate_labels=item["labels"])
    print(f"\n  Text: '{item['text']}'")
    print(f"  Labels & Scores:")
    for label, score in zip(result["labels"], result["scores"]):
        bar = "█" * int(score * 30)
        print(f"    {label:>12}: {score:.4f}  {bar}")

# You can even use custom, specific labels
print(f"\n  Custom labels (any text works):")
result = zero_shot(
    "My GPU keeps crashing when training the model",
    candidate_labels=["hardware issue", "software bug", "data problem", "model architecture"],
)
for label, score in zip(result["labels"], result["scores"]):
    bar = "█" * int(score * 30)
    print(f"    {label:>20}: {score:.4f}  {bar}")

# HOW TO VERIFY:
#   - Highest score matches the obvious category
#   - Scores sum to ~1.0 (probability distribution)
#   - Works with ANY labels you define — no retraining needed
#   - Less accurate than fine-tuned model, but zero effort


# ============================================================
# PART 5: Text Generation — GPT-Style Completion
# ============================================================
# WHAT IT DOES:
#   Input:  a prompt (start of text)
#   Output: continued text (the model writes what comes next)
#
# WHAT'S HAPPENING INSIDE:
#   Uses a DECODER-ONLY model (GPT-2):
#   1. Tokenize the prompt
#   2. Model predicts probability of NEXT token
#   3. Pick the most likely token (or sample from distribution)
#   4. Add that token to input, repeat
#
#   This is AUTOREGRESSIVE generation — one token at a time,
#   each new token conditioned on all previous tokens.
#   This is why GPT uses MASKED attention (Step 3, Part 10).

print("\n" + "=" * 60)
print("PART 5: Text Generation (GPT-2)")
print("=" * 60)

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "The future of artificial intelligence is",
    "In a surprising turn of events, scientists discovered",
    "Machine learning engineers should focus on",
]

for prompt in prompts:
    result = generator(
        prompt,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
    )
    generated = result[0]["generated_text"]
    print(f"\n  Prompt:    '{prompt}'")
    print(f"  Generated: '{generated}'")

# GENERATION PARAMETERS:
print(f"""
  Key parameters:
    max_new_tokens: how many tokens to generate (longer = slower)
    temperature:    controls randomness
                    0.1 = very focused/repetitive
                    0.7 = balanced (good default)
                    1.5 = very creative/chaotic
    do_sample:      True = sample from distribution (creative)
                    False = always pick most likely token (deterministic)
    top_k:          only consider top K most likely tokens
    top_p:          only consider tokens with cumulative prob < p
""")

# Show temperature effect
print(f"  Temperature effect (same prompt, different temps):")
prompt = "The meaning of life is"
for temp in [0.1, 0.7, 1.5]:
    result = generator(prompt, max_new_tokens=20, temperature=temp, do_sample=True)
    text = result[0]["generated_text"]
    print(f"    temp={temp}: '{text}'")


# ============================================================
# PART 6: Domain-Specific Models — Why Defaults Fail
# ============================================================
# pipeline() picks a default model, but you can specify any model
# from HuggingFace Hub (200,000+ models).
#
# WHY THIS MATTERS:
#   Default sentiment model = trained on movie reviews
#   "Revenue declined 15%" → it might say NEGATIVE
#   But is that financial-negative or just movie-review-negative?
#
#   Domain-specific models understand domain-specific language.
#   FinBERT knows "revenue declined" is bad for investors.
#   Default BERT only knows "declined" sounds negative in reviews.

print("\n" + "=" * 60)
print("PART 6: Default vs Domain-Specific Models")
print("=" * 60)

financial_texts = [
    "Revenue declined 15% due to supply chain disruptions",
    "The company announced a major acquisition worth $2 billion",
    "Profit margins expanded despite challenging market conditions",
]

# Default model (trained on movie reviews)
default_sentiment = pipeline("sentiment-analysis")
# Financial model (trained on financial text)
fin_sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")

print(f"\n  Comparing Default model vs FinBERT on financial text:\n")
for text in financial_texts:
    default_result = default_sentiment(text)[0]
    fin_result = fin_sentiment(text)[0]
    print(f"  Text: '{text}'")
    print(f"    Default (movie):     {default_result['label']:>8} ({default_result['score']:.3f})")
    print(f"    FinBERT (financial): {fin_result['label']:>8} ({fin_result['score']:.3f})")
    print()

print(f"""  → Default model often misclassifies financial sentiment
    because "declined" sounds negative in movie reviews too.
  → FinBERT understands financial context — much more accurate.
  → If no domain model exists, you fine-tune your own (Step 6).
""")


# ============================================================
# PART 7: Feature Extraction — Get Embeddings for Anything
# ============================================================
# WHAT IT DOES:
#   Returns the raw 768-dim vectors for each token.
#   No classification, no labels — just embeddings.
#
# WHY USEFUL:
#   - Semantic search (find similar documents)
#   - Clustering (group similar texts)
#   - Input to your own custom classifier
#   - Sentence similarity (compare two pieces of text)
#
# This is the bridge between "using pipeline" and "building custom systems."

print("=" * 60)
print("PART 7: Feature Extraction (Raw Embeddings)")
print("=" * 60)

import torch

feature_extractor = pipeline("feature-extraction")

texts = [
    "I love machine learning",
    "I enjoy deep learning",
    "The weather is sunny today",
]

# Get embeddings
embeddings = []
for text in texts:
    features = feature_extractor(text)
    # features is (1, seq_len, 768) — take mean across tokens for sentence embedding
    tensor = torch.tensor(features[0])
    sentence_emb = tensor.mean(dim=0)  # average all token vectors
    embeddings.append(sentence_emb)
    print(f"  '{text}' → shape {tensor.shape} → sentence emb: {sentence_emb.shape}")

# Compute similarity
from torch.nn.functional import cosine_similarity

print(f"\n  Sentence similarities:")
pairs = [(0, 1), (0, 2), (1, 2)]
for i, j in pairs:
    sim = cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
    print(f"    '{texts[i]}' vs")
    print(f"    '{texts[j]}'")
    print(f"    → similarity: {sim:.4f}\n")

# HOW TO VERIFY:
#   - Similar meaning sentences → high similarity
#   - Different meaning sentences → low similarity
#   - "love ML" vs "enjoy DL" should be higher than vs "weather is sunny"


# ============================================================
# PART 8: When to Use Pipeline vs Fine-Tune
# ============================================================

print("=" * 60)
print("PART 8: Pipeline vs Fine-Tuning — Decision Guide")
print("=" * 60)

print("""
  USE PIPELINE (pretrained, no training) WHEN:
  ┌──────────────────────────────────────────────────────────┐
  │  ✓ Task is generic (general sentiment, common NER)      │
  │  ✓ A domain-specific model already exists on HF Hub     │
  │  ✓ You're prototyping / building an MVP                 │
  │  ✓ You have NO labeled training data                    │
  │  ✓ Accuracy of 80-90% is acceptable                     │
  └──────────────────────────────────────────────────────────┘

  FINE-TUNE (Step 6) WHEN:
  ┌──────────────────────────────────────────────────────────┐
  │  ✓ Your domain is specific (legal, medical, financial)  │
  │  ✓ No existing model fits your exact task               │
  │  ✓ You HAVE labeled training data                       │
  │  ✓ You need accuracy > 90%                              │
  │  ✓ Your labels are custom (not standard categories)     │
  └──────────────────────────────────────────────────────────┘

  YOUR PROJECTS:
    FinancialNewsSentiment → fine-tune (domain-specific labels)
    ToxicContentDetection  → start with pipeline, fine-tune for accuracy
    CreditDefaultPrediction → not NLP (tabular data, different approach)
""")

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("SUMMARY: What You Now Know")
print("=" * 60)
print("""
  TASKS YOU CAN DO WITH ONE LINE OF CODE:
    pipeline("sentiment-analysis")       → positive/negative
    pipeline("ner")                      → extract entities
    pipeline("fill-mask")               → predict missing words
    pipeline("zero-shot-classification") → classify anything
    pipeline("text-generation")          → GPT-style completion
    pipeline("feature-extraction")       → get raw embeddings

  KEY CONCEPTS:
    1. pipeline() wraps tokenizer + model + post-processing
    2. Default models work for generic tasks
    3. Specify model= for domain-specific needs (e.g., FinBERT)
    4. Zero-shot = classify without ANY training data
    5. Generation parameters control creativity vs focus
    6. Feature extraction = bridge to custom systems
    7. Know when to use as-is vs when to fine-tune

  THE FLOW SO FAR:
    Step 1: Tokenization        (text → numbers)
    Step 2: Embeddings          (numbers → vectors)
    Step 3: Self-Attention      (vectors → context-aware vectors)
    Step 4: Transformer Block   (the complete architecture)
    Step 5: Inference Pipeline  (USE models for real tasks) ← YOU ARE HERE

  NEXT STEP: 06_fine_tuning.py
    Take a pretrained model and train it on YOUR data.
    This is how you build production NLP systems.
""")

"""
STEP 1: Tokenization — See What the Model Actually Receives
============================================================

THE BIG PICTURE:
    You type: "I love NLP"
    Model sees: [101, 1045, 2293, 17953, 2361, 102]

    Models don't understand text. They understand numbers.
    Tokenization is the bridge: Text → Numbers (token IDs)

    Every single thing you do in NLP starts here.
    If your tokenization is wrong, everything downstream is wrong.

WHAT YOU'LL LEARN:
    1. How text becomes tokens (subword splitting)
    2. What input_ids are (the actual numbers fed to the model)
    3. What attention_mask is (tells model what to pay attention to)
    4. What special tokens are ([CLS], [SEP], [PAD])
    5. How padding and truncation work (batching multiple sentences)
"""

from transformers import AutoTokenizer

# ============================================================
# PART 1: Basic Tokenization
# ============================================================
# WHY AutoTokenizer?
#   Each model has its OWN tokenizer trained on its OWN data.
#   BERT tokenizer != GPT tokenizer != T5 tokenizer.
#   AutoTokenizer picks the right one for the model you specify.

print("=" * 60)
print("PART 1: Basic Tokenization")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love Natural Language Processing"

# Step 1: See the tokens (human readable)
tokens = tokenizer.tokenize(text)
print(f"\nOriginal text: '{text}'")
print(f"Tokens:        {tokens}")

# Step 2: Convert tokens to IDs (what model actually sees)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs:     {token_ids}")

# Step 3: Decode back to text (verify round-trip)
decoded = tokenizer.decode(token_ids)
print(f"Decoded back:  '{decoded}'")

# HOW TO VERIFY:
#   - Token count >= word count (words get split into subwords)
#   - Decoded text should match original (minus casing for uncased models)
#   - No [UNK] tokens means the tokenizer handled everything

print(f"\n✓ Word count: {len(text.split())} | Token count: {len(tokens)}")
print(f"✓ Round-trip match: {decoded.strip() == text.lower()}")


# ============================================================
# PART 2: What is Subword Tokenization?
# ============================================================
# WHY SUBWORDS?
#   Problem: vocabulary can't contain every possible word.
#   Solution: break rare/unknown words into smaller pieces.
#
#   "unhappiness" → ["un", "##happ", "##iness"]
#   "##" prefix means "this continues the previous piece"
#
#   This is WHY models can handle words they've never seen before.

print("\n" + "=" * 60)
print("PART 2: Subword Tokenization")
print("=" * 60)

examples = [
    "unhappiness",       # common word, might split
    "tokenization",      # NLP term
    "Mohit",             # proper noun — likely splits
    "transformers",      # common in this context
    "abcxyz",            # gibberish — will split heavily
]

for word in examples:
    tokens = tokenizer.tokenize(word)
    print(f"  '{word}' → {tokens}")

# HOW TO VERIFY:
#   - Common words = fewer tokens (1-2)
#   - Rare/made-up words = many tokens (heavy splitting)
#   - "##" prefix means subword continuation
#   - No word should produce [UNK] unless truly unknown


# ============================================================
# PART 3: The Full Encoding (what you actually pass to a model)
# ============================================================
# WHY encoded = tokenizer(text)?
#   tokenizer.tokenize() only splits text.
#   tokenizer() does EVERYTHING:
#     1. Splits into tokens
#     2. Adds special tokens ([CLS] at start, [SEP] at end)
#     3. Converts to IDs
#     4. Creates attention mask
#     5. Returns PyTorch tensors (if you ask)
#
# SPECIAL TOKENS:
#   [CLS] (id=101) — "Classification" token. Added at the START.
#                     The model's representation of this token becomes
#                     a summary of the entire sentence.
#   [SEP] (id=102) — "Separator" token. Added at the END.
#                     Marks sentence boundaries.
#   [PAD] (id=0)   — "Padding" token. Used to make sentences equal length.

print("\n" + "=" * 60)
print("PART 3: Full Encoding")
print("=" * 60)

text = "I love NLP"
encoded = tokenizer(text, return_tensors="pt")

print(f"\nText: '{text}'")
print(f"Input IDs:      {encoded['input_ids']}")
print(f"Attention Mask:  {encoded['attention_mask']}")

# Decode to see what those IDs represent
all_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
print(f"Tokens with special: {all_tokens}")

# HOW TO VERIFY:
#   - First token should be [CLS] (id=101)
#   - Last token should be [SEP] (id=102)
#   - Attention mask should be all 1s (no padding yet)
#   - Length = original tokens + 2 (for [CLS] and [SEP])

print(f"\n✓ Starts with [CLS]: {all_tokens[0] == '[CLS]'}")
print(f"✓ Ends with [SEP]:   {all_tokens[-1] == '[SEP]'}")
print(f"✓ All attention = 1:  {all(encoded['attention_mask'][0].tolist())}")


# ============================================================
# PART 4: Attention Mask — WHY it matters
# ============================================================
# WHAT IS IT?
#   attention_mask = [1, 1, 1, 0, 0]
#   1 = "real token, pay attention to this"
#   0 = "padding, ignore this"
#
# WHY DO WE NEED IT?
#   Models need fixed-size input. But sentences have different lengths.
#   Short sentences get padded with [PAD] tokens.
#   Without the mask, the model would "read" the padding as real text.

print("\n" + "=" * 60)
print("PART 4: Attention Mask & Padding")
print("=" * 60)

sentences = [
    "I love NLP",
    "Transformers changed the world of deep learning",
]

# Without padding
for s in sentences:
    enc = tokenizer(s)
    print(f"  '{s}' → length {len(enc['input_ids'])}")

# With padding — makes both sentences the same length
batch = tokenizer(sentences, padding=True, return_tensors="pt")

print(f"\nAfter padding (both same length):")
for i, s in enumerate(sentences):
    ids = batch["input_ids"][i].tolist()
    mask = batch["attention_mask"][i].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"\n  Sentence {i+1}: '{s}'")
    print(f"  Tokens: {tokens}")
    print(f"  IDs:    {ids}")
    print(f"  Mask:   {mask}")

# HOW TO VERIFY:
#   - Both sentences have SAME length after padding
#   - Shorter sentence has [PAD] tokens at the end
#   - [PAD] positions have 0 in attention_mask
#   - Longer sentence has all 1s in attention_mask

pad_count = batch["attention_mask"][0].tolist().count(0)
print(f"\n✓ Both same length: {batch['input_ids'].shape}")
print(f"✓ Sentence 1 has {pad_count} padding tokens")


# ============================================================
# PART 5: Truncation — handling long text
# ============================================================
# WHY TRUNCATION?
#   BERT has a max length of 512 tokens.
#   If your text is longer, you MUST truncate or it errors out.
#   In real projects, this is a common source of bugs.

print("\n" + "=" * 60)
print("PART 5: Truncation")
print("=" * 60)

long_text = "I love NLP. " * 100  # artificially long

encoded_no_trunc = tokenizer(long_text)
encoded_trunc = tokenizer(long_text, max_length=20, truncation=True)

print(f"Original token count:  {len(encoded_no_trunc['input_ids'])}")
print(f"After truncation (20): {len(encoded_trunc['input_ids'])}")

tokens_trunc = tokenizer.convert_ids_to_tokens(encoded_trunc["input_ids"])
print(f"Truncated tokens: {tokens_trunc}")

# HOW TO VERIFY:
#   - Truncated length = max_length exactly
#   - Still has [CLS] at start and [SEP] at end
#   - Content is cut, not corrupted

print(f"\n✓ Length = max_length: {len(encoded_trunc['input_ids']) == 20}")
print(f"✓ Starts with [CLS]:  {tokens_trunc[0] == '[CLS]'}")
print(f"✓ Ends with [SEP]:    {tokens_trunc[-1] == '[SEP]'}")


# ============================================================
# PART 6: Different Tokenizers for Different Models
# ============================================================
# WHY THIS MATTERS:
#   Same text → different tokens depending on the model.
#   BERT uses WordPiece tokenization.
#   GPT-2 uses BPE (Byte Pair Encoding).
#   You MUST use the tokenizer that matches your model.

print("\n" + "=" * 60)
print("PART 6: BERT vs GPT-2 Tokenizers")
print("=" * 60)

tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2")

text = "Tokenization is interesting"

bert_tokens = tokenizer.tokenize(text)
gpt_tokens = tokenizer_gpt.tokenize(text)

print(f"\nText: '{text}'")
print(f"BERT tokens: {bert_tokens}  (count: {len(bert_tokens)})")
print(f"GPT2 tokens: {gpt_tokens}  (count: {len(gpt_tokens)})")

# KEY DIFFERENCES:
#   - BERT subwords use "##" prefix
#   - GPT-2 subwords use "Ġ" prefix (space before word)
#   - BERT adds [CLS]/[SEP], GPT-2 doesn't by default
#   - They split words differently

print("\nBERT special tokens:", tokenizer.all_special_tokens)
print("GPT2 special tokens:", tokenizer_gpt.all_special_tokens)


# ============================================================
# PART 7: The Vocabulary — where IDs come from
# ============================================================
# Every tokenizer has a fixed vocabulary: a dictionary mapping
# tokens → IDs. This was built during tokenizer training.
# The model's embedding layer has exactly vocab_size rows,
# one for each token.

print("\n" + "=" * 60)
print("PART 7: Vocabulary")
print("=" * 60)

print(f"BERT vocab size: {tokenizer.vocab_size}")
print(f"GPT2 vocab size: {tokenizer_gpt.vocab_size}")

# Look up specific tokens
for word in ["hello", "the", "[CLS]", "[PAD]"]:
    token_id = tokenizer.convert_tokens_to_ids(word)
    print(f"  '{word}' → ID {token_id}")

# [UNK] = unknown token (ID 100 in BERT)
# If a subword isn't in vocabulary, it becomes [UNK]
print(f"\n[UNK] token ID: {tokenizer.convert_tokens_to_ids('[UNK]')}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: What You Now Know")
print("=" * 60)
print("""
1. tokenizer.tokenize(text)  → splits text into subword tokens
2. tokenizer(text)           → full encoding (IDs + mask + special tokens)
3. input_ids                 → the numbers the model actually processes
4. attention_mask            → 1 for real tokens, 0 for padding
5. [CLS], [SEP], [PAD]      → special tokens with specific roles
6. padding + truncation      → how to handle variable-length inputs
7. Each model needs its OWN tokenizer — never mix them

NEXT STEP: Run this file, read each output, then move to
           02_embeddings.py to see how these IDs become vectors.
""")

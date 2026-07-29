"""
STEP 2: Embeddings — How Token IDs Become Vectors
===================================================

THE BIG PICTURE:
    Step 1 gave us: [101, 1045, 2293, 102]   (just numbers, no meaning)
    Step 2 gives us: each ID → a 768-dim vector (rich with meaning)

    Token ID 2293 ("love") is just an index.
    But its embedding vector [0.02, -0.15, 0.44, ...] captures MEANING.
    Similar words have similar vectors. That's the magic.

WHY EMBEDDINGS EXIST:
    ID 2293 ("love") and ID 2293+1 ("loved"?) — consecutive IDs
    don't mean similar things. IDs are arbitrary dictionary indices.

    Embeddings fix this: "love" and "adore" will have NEARBY vectors
    even though their IDs are far apart.

WHAT YOU'LL LEARN:
    1. Word embeddings — meaning of each token
    2. Positional embeddings — where each token sits in the sentence
    3. How they combine — word + position = what model actually uses
    4. Why similar words have similar vectors (cosine similarity)
    5. How to inspect and verify all of this
"""

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# ============================================================
# PART 1: The Embedding Layer — A Giant Lookup Table
# ============================================================
# WHAT IS IT?
#   A matrix of shape (vocab_size, hidden_size) = (30522, 768)
#   Row 0    = vector for [PAD]
#   Row 101  = vector for [CLS]
#   Row 2293 = vector for "love"
#   ...
#   Each row is 768 numbers that represent that token's meaning.
#
# HOW IT WORKS:
#   Input:  token ID 2293
#   Output: row 2293 from the matrix → a 768-dim vector
#   That's it. It's just a table lookup. No math yet.
#
# WHERE DO THESE VECTORS COME FROM?
#   They were LEARNED during BERT's pretraining on massive text data.
#   The model adjusted these vectors so that semantically similar
#   words ended up with similar vectors.

print("=" * 60)
print("PART 1: The Embedding Layer Structure")
print("=" * 60)

embeddings = model.embeddings
print(f"\nBERT's embedding layers:")
print(f"  Word embeddings:     {embeddings.word_embeddings}")
print(f"  Position embeddings: {embeddings.position_embeddings}")
print(f"  Token type embeddings: {embeddings.token_type_embeddings}")

word_emb_weight = embeddings.word_embeddings.weight
print(f"\nWord embedding matrix shape: {word_emb_weight.shape}")
print(f"  → {word_emb_weight.shape[0]} tokens in vocabulary")
print(f"  → {word_emb_weight.shape[1]} dimensions per token")

# HOW TO VERIFY:
#   - Shape should be (30522, 768) for bert-base
#   - 30522 = vocab size (from step 1)
#   - 768 = BERT-base hidden size

print(f"\n✓ Vocab size matches: {word_emb_weight.shape[0] == tokenizer.vocab_size}")
print(f"✓ Hidden size is 768: {word_emb_weight.shape[1] == 768}")


# ============================================================
# PART 2: Word Embeddings — Looking Up Meaning
# ============================================================
# For each token ID, we grab its row from the embedding matrix.
# This row IS the token's meaning — as a vector of 768 numbers.

print("\n" + "=" * 60)
print("PART 2: Word Embeddings (Token → Vector)")
print("=" * 60)

text = "I love NLP"
encoded = tokenizer(text, return_tensors="pt")
input_ids = encoded["input_ids"][0]

tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"\nText: '{text}'")
print(f"Tokens: {tokens}")
print(f"IDs:    {input_ids.tolist()}")

# Manually look up each token's embedding
word_embeddings = embeddings.word_embeddings(input_ids.unsqueeze(0))
# unsqueeze adds batch dimension: (6,) → (1, 6) → output (1, 6, 768)

print(f"\nWord embeddings shape: {word_embeddings.shape}")
print(f"  → batch_size=1, sequence_length={word_embeddings.shape[1]}, "
      f"embedding_dim={word_embeddings.shape[2]}")

# Show first 10 values of each token's embedding
print(f"\nFirst 10 values of each token's embedding:")
for i, token in enumerate(tokens):
    vec = word_embeddings[0, i, :10].detach().tolist()
    vec_str = [f"{v:+.4f}" for v in vec]
    print(f"  {token:>15} → [{', '.join(vec_str)}, ...]")

# HOW TO VERIFY:
#   - Each token maps to a 768-dim vector
#   - Different tokens → different vectors
#   - Same token always → same vector (it's a fixed lookup)

love_vec = word_embeddings[0, 2]  # "love" is at position 2
love_again = embeddings.word_embeddings(torch.tensor([2293]))  # ID for "love"
print(f"\n✓ Same token same vector: {torch.allclose(love_vec, love_again[0], atol=1e-6)}")


# ============================================================
# PART 3: Positional Embeddings — Where Am I in the Sentence?
# ============================================================
# WHY NEEDED?
#   Word embeddings don't know word ORDER.
#   "dog bites man" and "man bites dog" would have the same
#   word embeddings (just rearranged). But they mean different things!
#
# HOW IT WORKS:
#   Position 0 → a 768-dim vector
#   Position 1 → a different 768-dim vector
#   Position 2 → another different 768-dim vector
#   ...up to position 511 (BERT's max length)
#
#   These are also LEARNED during pretraining, not hand-designed.
#   (Original transformer paper used sin/cos formulas instead,
#    but BERT learns them from data.)

print("\n" + "=" * 60)
print("PART 3: Positional Embeddings")
print("=" * 60)

pos_emb_weight = embeddings.position_embeddings.weight
print(f"Position embedding matrix shape: {pos_emb_weight.shape}")
print(f"  → {pos_emb_weight.shape[0]} max positions (0 to {pos_emb_weight.shape[0]-1})")
print(f"  → {pos_emb_weight.shape[1]} dimensions per position")

# Get positional embeddings for our sentence
seq_length = input_ids.shape[0]
position_ids = torch.arange(seq_length).unsqueeze(0)  # [0, 1, 2, 3, 4, 5]
pos_embeddings = embeddings.position_embeddings(position_ids)

print(f"\nPosition IDs: {position_ids.tolist()}")
print(f"Position embeddings shape: {pos_embeddings.shape}")

# Show first 10 values
print(f"\nFirst 10 values of each position's embedding:")
for i in range(seq_length):
    vec = pos_embeddings[0, i, :10].detach().tolist()
    vec_str = [f"{v:+.4f}" for v in vec]
    print(f"  Position {i} ({tokens[i]:>15}) → [{', '.join(vec_str)}, ...]")

# KEY INSIGHT: nearby positions have similar embeddings
from torch.nn.functional import cosine_similarity

sim_0_1 = cosine_similarity(pos_emb_weight[0].unsqueeze(0),
                             pos_emb_weight[1].unsqueeze(0)).item()
sim_0_100 = cosine_similarity(pos_emb_weight[0].unsqueeze(0),
                               pos_emb_weight[100].unsqueeze(0)).item()
sim_0_500 = cosine_similarity(pos_emb_weight[0].unsqueeze(0),
                               pos_emb_weight[500].unsqueeze(0)).item()

print(f"\nPosition similarity (cosine):")
print(f"  Position 0 vs 1:   {sim_0_1:.4f}  (nearby → similar)")
print(f"  Position 0 vs 100: {sim_0_100:.4f}  (far → less similar)")
print(f"  Position 0 vs 500: {sim_0_500:.4f}  (very far → even less)")

# HOW TO VERIFY:
#   - Shape is (512, 768) — 512 positions, 768 dims
#   - Nearby positions should have higher cosine similarity
#   - Same word at different positions → different final embeddings


# ============================================================
# PART 4: Combining — Word + Position = Final Embedding
# ============================================================
# THE FORMULA:
#   final_embedding = word_embedding + position_embedding + token_type_embedding
#   then → LayerNorm → Dropout
#
# WHY ADD THEM?
#   Adding combines the "what" (word) with the "where" (position).
#   "love" at position 2 ≠ "love" at position 5
#   because they get different positional vectors added.
#
# TOKEN TYPE EMBEDDING:
#   For tasks with two sentences (e.g., "Is sentence A related to B?")
#   Sentence A tokens get type 0, Sentence B tokens get type 1.
#   For single sentences, it's all 0s — you can mostly ignore it.

print("\n" + "=" * 60)
print("PART 4: Combining Word + Position Embeddings")
print("=" * 60)

# Do it manually step by step
token_type_ids = torch.zeros_like(input_ids).unsqueeze(0)
token_type_embeddings = embeddings.token_type_embeddings(token_type_ids)

# Manual combination
combined = word_embeddings + pos_embeddings + token_type_embeddings
print(f"\nWord embeddings shape:       {word_embeddings.shape}")
print(f"Position embeddings shape:   {pos_embeddings.shape}")
print(f"Token type embeddings shape: {token_type_embeddings.shape}")
print(f"Combined shape:              {combined.shape}")

# Compare with what BERT actually produces
with torch.no_grad():
    actual_output = embeddings(input_ids.unsqueeze(0))

# They won't be exactly equal because of LayerNorm + Dropout
# But the pre-norm values should match
pre_norm = word_embeddings + pos_embeddings + token_type_embeddings
normed = embeddings.LayerNorm(pre_norm)

print(f"\n✓ After LayerNorm matches BERT: "
      f"{torch.allclose(normed, actual_output, atol=1e-6)}")

print(f"\nWhat happens at each step (showing dimension 0 for 'love'):")
w = word_embeddings[0, 2, 0].item()
p = pos_embeddings[0, 2, 0].item()
t = token_type_embeddings[0, 2, 0].item()
print(f"  Word embedding[0]:       {w:+.4f}")
print(f"  Position embedding[0]:   {p:+.4f}")
print(f"  Token type embedding[0]: {t:+.4f}")
print(f"  Sum:                     {w+p+t:+.4f}")
print(f"  After LayerNorm:         {actual_output[0, 2, 0].item():+.4f}")


# ============================================================
# PART 5: Similar Words Have Similar Embeddings
# ============================================================
# This is the PROOF that embeddings capture meaning.
# "king" and "queen" should be closer to each other
# than "king" and "banana".

print("\n" + "=" * 60)
print("PART 5: Semantic Similarity in Embeddings")
print("=" * 60)

def get_word_embedding(word):
    token_id = tokenizer.convert_tokens_to_ids(word)
    if token_id == tokenizer.unk_token_id:
        return None, token_id
    return embeddings.word_embeddings.weight[token_id].detach(), token_id

word_pairs = [
    ("king", "queen"),      # semantically related
    ("king", "prince"),     # semantically related
    ("king", "banana"),     # unrelated
    ("happy", "sad"),       # related (both emotions) but opposite
    ("happy", "joyful"),    # very similar meaning
    ("happy", "computer"),  # unrelated
    ("dog", "cat"),         # related (both animals)
    ("dog", "algorithm"),   # unrelated
]

print(f"\nCosine similarity between word embeddings:")
print(f"  (1.0 = identical, 0.0 = unrelated, -1.0 = opposite)\n")

for w1, w2 in word_pairs:
    emb1, id1 = get_word_embedding(w1)
    emb2, id2 = get_word_embedding(w2)
    if emb1 is not None and emb2 is not None:
        sim = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        bar = "█" * int(abs(sim) * 20)
        print(f"  {w1:>10} vs {w2:<10} → {sim:+.4f}  {bar}")

# HOW TO VERIFY:
#   - Related words (king/queen) should have higher similarity
#   - Unrelated words (king/banana) should have lower similarity
#   - This is NOT perfect — BERT embeddings are pretrained general-purpose
#   - After fine-tuning, these relationships get even better for your task


# ============================================================
# PART 6: Same Word, Different Position = Different Embedding
# ============================================================
# This proves positional embeddings actually change things.

print("\n" + "=" * 60)
print("PART 6: Position Changes the Embedding")
print("=" * 60)

text1 = "the cat sat on the mat"
encoded1 = tokenizer(text1, return_tensors="pt")

with torch.no_grad():
    output1 = embeddings(encoded1["input_ids"])

tokens1 = tokenizer.convert_ids_to_tokens(encoded1["input_ids"][0])

# "the" appears at position 1 and position 5
the_positions = [i for i, t in enumerate(tokens1) if t == "the"]
print(f"\nText: '{text1}'")
print(f"Tokens: {tokens1}")
print(f"'the' appears at positions: {the_positions}")

if len(the_positions) >= 2:
    pos_a, pos_b = the_positions[0], the_positions[1]

    # Word embeddings are the same (same word)
    word_a = embeddings.word_embeddings(encoded1["input_ids"][0, pos_a:pos_a+1])
    word_b = embeddings.word_embeddings(encoded1["input_ids"][0, pos_b:pos_b+1])
    word_sim = cosine_similarity(word_a, word_b).item()

    # Final embeddings are different (different positions)
    final_a = output1[0, pos_a]
    final_b = output1[0, pos_b]
    final_sim = cosine_similarity(final_a.unsqueeze(0), final_b.unsqueeze(0)).item()

    print(f"\n'the' at position {pos_a} vs 'the' at position {pos_b}:")
    print(f"  Word embedding similarity:  {word_sim:.4f}  (same word → identical)")
    print(f"  Final embedding similarity: {final_sim:.4f}  (different position → different!)")

# HOW TO VERIFY:
#   - Word embeddings for same word = identical (similarity 1.0)
#   - Final embeddings for same word at different positions ≠ identical
#   - This is proof that positional information is encoded


# ============================================================
# PART 7: Embedding vs Contextual Representation
# ============================================================
# CRITICAL DISTINCTION:
#   What we've seen so far = STATIC embeddings (input to layer 1)
#   What BERT produces = CONTEXTUAL embeddings (output of all layers)
#
#   Static:      "bank" always has the same embedding
#   Contextual:  "river bank" vs "bank account" → different vectors
#
#   The embedding layer is just the STARTING POINT.
#   The transformer layers (attention) make it context-aware.

print("\n" + "=" * 60)
print("PART 7: Static vs Contextual Embeddings")
print("=" * 60)

texts = [
    "I went to the river bank to fish",
    "I went to the bank to deposit money",
]

for text in texts:
    encoded = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    bank_pos = tokens.index("bank")

    # Static embedding (from embedding layer only)
    static_emb = embeddings.word_embeddings(encoded["input_ids"][0, bank_pos:bank_pos+1])

    # Contextual embedding (after all transformer layers)
    with torch.no_grad():
        full_output = model(**encoded)
    contextual_emb = full_output.last_hidden_state[0, bank_pos:bank_pos+1]

    print(f"\n  Text: '{text}'")
    print(f"  'bank' at position {bank_pos}")
    print(f"  Static embedding (first 5):     "
          f"{[f'{v:.4f}' for v in static_emb[0, :5].tolist()]}")
    print(f"  Contextual embedding (first 5): "
          f"{[f'{v:.4f}' for v in contextual_emb[0, :5].tolist()]}")

# Compare the contextual embeddings of "bank" in both sentences
enc1 = tokenizer(texts[0], return_tensors="pt")
enc2 = tokenizer(texts[1], return_tensors="pt")
tok1 = tokenizer.convert_ids_to_tokens(enc1["input_ids"][0])
tok2 = tokenizer.convert_ids_to_tokens(enc2["input_ids"][0])

with torch.no_grad():
    out1 = model(**enc1)
    out2 = model(**enc2)

bank_pos1 = tok1.index("bank")
bank_pos2 = tok2.index("bank")

ctx1 = out1.last_hidden_state[0, bank_pos1]
ctx2 = out2.last_hidden_state[0, bank_pos2]

static_sim = 1.0  # same word → identical static embedding
ctx_sim = cosine_similarity(ctx1.unsqueeze(0), ctx2.unsqueeze(0)).item()

print(f"\n  'bank' (river) vs 'bank' (money):")
print(f"  Static embedding similarity:     {static_sim:.4f}  (same word = identical)")
print(f"  Contextual embedding similarity: {ctx_sim:.4f}  (different context = different!)")
print(f"\n  → This is what the transformer layers do.")
print(f"    They take static embeddings and make them context-aware.")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: What You Now Know")
print("=" * 60)
print("""
1. Word embeddings:  token ID → 768-dim vector (lookup table)
                     Shape: (30522, 768) for BERT-base
2. Positional embeddings: position index → 768-dim vector
                     Shape: (512, 768) — max 512 positions
3. They ADD together: word + position + token_type = input embedding
4. Similar words → similar vectors (cosine similarity proves it)
5. Same word, different position → different final embedding
6. These are STATIC embeddings — just the starting point
7. Transformer layers turn them into CONTEXTUAL embeddings
   where "bank" means different things in different sentences

THE FLOW SO FAR:
  Text → Tokenizer → IDs → Embedding Layer → Vectors (768-dim)
       (Step 1)            (Step 2 - you are here)

NEXT STEP: 03_self_attention.py — how the model looks at ALL
           tokens to understand context (the transformer's core).
""")

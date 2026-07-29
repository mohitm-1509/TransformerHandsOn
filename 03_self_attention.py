"""
STEP 3: Self-Attention From Scratch
=====================================

THE BIG PICTURE:
    Step 2 gave us: each token = a 768-dim vector (static, context-free)
    Step 3 does:    each token LOOKS AT every other token to understand context

    "I saw a bank near the river"
    When processing "bank", self-attention lets the model look at
    "river" and think "ah, this bank means riverbank, not financial bank."

    This is THE mechanism that makes transformers powerful.
    Everything else (feedforward layers, normalization) is supporting cast.

HOW IT WORKS (4 steps):
    1. Each token creates 3 vectors: Query (Q), Key (K), Value (V)
    2. Q·K^T = attention scores (who should I pay attention to?)
    3. Softmax → attention weights (normalized to sum to 1)
    4. Weights × V = new representation (context-aware embedding)

    Think of it as: "I'm token X (Query). Which other tokens (Keys)
    are relevant to me? Let me take a weighted mix of their info (Values)."

WHAT YOU'LL LEARN:
    1. Code Q, K, V projections from scratch
    2. Compute attention scores manually
    3. Why we scale by sqrt(d_k)
    4. How softmax turns scores into weights
    5. Multi-head attention — why and how
    6. Compare your scratch code with BERT's actual attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")


# ============================================================
# PART 1: Q, K, V — The Three Roles Each Token Plays
# ============================================================
# WHY THREE VECTORS?
#   Each token needs to play 3 different roles:
#
#   Query (Q): "What am I looking for?"
#       → When processing "bank", Q encodes "I need context clues"
#
#   Key (K): "What do I offer?"
#       → "river" advertises "I'm a nature word"
#       → "money" advertises "I'm a finance word"
#
#   Value (V): "What information do I carry?"
#       → The actual content that gets passed forward
#
#   Q and K are used to compute ATTENTION SCORES (who matches whom).
#   V is the actual information that flows based on those scores.
#
# WHY NOT JUST USE THE EMBEDDING DIRECTLY?
#   The raw embedding captures what the word IS.
#   Q, K, V are learned PROJECTIONS that capture different ROLES.
#   Same word might be a good Key for one context, good Query for another.
#   The projection matrices (W_Q, W_K, W_V) learn these role mappings.

print("=" * 60)
print("PART 1: Creating Q, K, V From Scratch")
print("=" * 60)

# Let's work with a simple sentence
text = "The cat sat on the mat"
encoded = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

# Get embeddings (output of Step 2)
with torch.no_grad():
    embeddings_output = model.embeddings(encoded["input_ids"])

seq_len = embeddings_output.shape[1]
d_model = embeddings_output.shape[2]  # 768

print(f"\nText: '{text}'")
print(f"Tokens: {tokens}")
print(f"Embedding shape: {embeddings_output.shape}  (batch=1, seq={seq_len}, dim={d_model})")

# Create Q, K, V projection matrices FROM SCRATCH
# These are simple linear layers (matrix multiply, no bias for clarity)
d_k = 64  # dimension of each Q/K/V vector (BERT uses 64 per head)

torch.manual_seed(42)
W_Q = nn.Linear(d_model, d_k, bias=False)
W_K = nn.Linear(d_model, d_k, bias=False)
W_V = nn.Linear(d_model, d_k, bias=False)

# Project embeddings into Q, K, V
with torch.no_grad():
    Q = W_Q(embeddings_output)  # (1, seq_len, 64)
    K = W_K(embeddings_output)  # (1, seq_len, 64)
    V = W_V(embeddings_output)  # (1, seq_len, 64)

print(f"\nProjection: {d_model}-dim embedding → {d_k}-dim Q/K/V")
print(f"Q shape: {Q.shape}  (each token's query)")
print(f"K shape: {K.shape}  (each token's key)")
print(f"V shape: {V.shape}  (each token's value)")

# HOW TO VERIFY:
#   - Q, K, V all have shape (1, seq_len, d_k)
#   - d_k = 64 (BERT-base uses 768/12 = 64 per head)
#   - Values are different for Q, K, V (different projection matrices)

print(f"\n✓ Q, K, V shapes match: {Q.shape == K.shape == V.shape}")
print(f"✓ Dimension reduced: {d_model} → {d_k}")


# ============================================================
# PART 2: Attention Scores — Who Should Pay Attention to Whom?
# ============================================================
# THE FORMULA:
#   scores = Q · K^T
#
# WHAT THIS DOES:
#   For EVERY pair of tokens, compute a dot product.
#   Q[i] · K[j] = "how much should token i attend to token j?"
#
#   High dot product → these tokens are relevant to each other
#   Low dot product  → these tokens are not relevant
#
#   Result is a (seq_len × seq_len) matrix:
#   Row i, Column j = "how much token i attends to token j"

print("\n" + "=" * 60)
print("PART 2: Computing Attention Scores (Q · K^T)")
print("=" * 60)

# Matrix multiply: (1, seq_len, d_k) × (1, d_k, seq_len) → (1, seq_len, seq_len)
scores = torch.matmul(Q, K.transpose(-2, -1))

print(f"\nAttention scores shape: {scores.shape}")
print(f"  → {seq_len} tokens × {seq_len} tokens = who attends to whom\n")

# Display the raw attention score matrix
print("Raw attention scores (each row = one token's attention to all others):")
print(f"{'':>10}", end="")
for t in tokens:
    print(f"{t:>10}", end="")
print()

for i, token in enumerate(tokens):
    print(f"{token:>10}", end="")
    for j in range(seq_len):
        print(f"{scores[0, i, j].item():>10.2f}", end="")
    print()

# HOW TO VERIFY:
#   - Shape is (1, seq_len, seq_len) — square matrix
#   - Matrix is NOT symmetric (Q·K^T ≠ K·Q^T because Q ≠ K)
#   - Diagonal values are not necessarily the highest


# ============================================================
# PART 3: Scaling — Why Divide by sqrt(d_k)?
# ============================================================
# THE PROBLEM:
#   When d_k is large (64), dot products can get very large.
#   Large values → softmax pushes everything to 0 or 1
#   → gradients vanish → model can't learn
#
# THE FIX:
#   Divide by sqrt(d_k) to keep values in a reasonable range.
#   scores = Q·K^T / sqrt(d_k)
#
# WHY sqrt(d_k)?
#   If Q and K elements are ~N(0,1), then Q·K has variance ≈ d_k.
#   Dividing by sqrt(d_k) brings variance back to ~1.
#   This is math, not a magic number.

print("\n" + "=" * 60)
print("PART 3: Scaling by sqrt(d_k)")
print("=" * 60)

import math

scale = math.sqrt(d_k)
scaled_scores = scores / scale

print(f"\nScale factor: sqrt({d_k}) = {scale:.2f}")
print(f"\nBefore scaling — score stats:")
print(f"  Mean: {scores.mean().item():.4f}  Std: {scores.std().item():.4f}")
print(f"  Min:  {scores.min().item():.4f}  Max: {scores.max().item():.4f}")
print(f"\nAfter scaling — score stats:")
print(f"  Mean: {scaled_scores.mean().item():.4f}  Std: {scaled_scores.std().item():.4f}")
print(f"  Min:  {scaled_scores.min().item():.4f}  Max: {scaled_scores.max().item():.4f}")

# Show the effect on softmax
unscaled_softmax = F.softmax(scores[0, 0], dim=-1)
scaled_softmax = F.softmax(scaled_scores[0, 0], dim=-1)

print(f"\nSoftmax of [CLS] row WITHOUT scaling: {[f'{v:.4f}' for v in unscaled_softmax.tolist()]}")
print(f"Softmax of [CLS] row WITH scaling:    {[f'{v:.4f}' for v in scaled_softmax.tolist()]}")
print(f"\n→ Without scaling: attention is peaky (one token dominates)")
print(f"→ With scaling: attention is smoother (can attend to multiple tokens)")

# HOW TO VERIFY:
#   - Scaled scores have smaller magnitude than raw scores
#   - Scaled softmax is smoother (more spread out)
#   - Unscaled softmax is peaky (one value close to 1, rest near 0)


# ============================================================
# PART 4: Softmax — Turning Scores into Weights
# ============================================================
# WHY SOFTMAX?
#   Raw scores can be any number: -5, 0, +10, etc.
#   We need WEIGHTS that:
#     1. Are all positive (negative attention doesn't make sense)
#     2. Sum to 1.0 per row (each token's attention must sum to 100%)
#   Softmax does exactly this.
#
# WHAT THE OUTPUT MEANS:
#   attention_weights[i][j] = "fraction of attention token i gives to token j"
#   Row i sums to 1.0 — it's a probability distribution.

print("\n" + "=" * 60)
print("PART 4: Softmax → Attention Weights")
print("=" * 60)

attention_weights = F.softmax(scaled_scores, dim=-1)

print(f"\nAttention weights shape: {attention_weights.shape}")
print(f"Each row sums to: {attention_weights[0].sum(dim=-1).tolist()}")

print(f"\nAttention weight matrix (each row sums to 1.0):")
print(f"{'':>10}", end="")
for t in tokens:
    print(f"{t:>10}", end="")
print()

for i, token in enumerate(tokens):
    print(f"{token:>10}", end="")
    for j in range(seq_len):
        w = attention_weights[0, i, j].item()
        print(f"{w:>10.4f}", end="")
    print()

# Visualize with bars
print(f"\nVisualized (who each token attends to most):")
for i, token in enumerate(tokens):
    weights = attention_weights[0, i].detach().tolist()
    max_j = weights.index(max(weights))
    bars = [("█" * int(w * 30)) for w in weights]
    print(f"\n  {token:>10} attends to:")
    for j, (t, bar, w) in enumerate(zip(tokens, bars, weights)):
        marker = " ◄ max" if j == max_j else ""
        if w > 0.05:
            print(f"    {t:>10} [{bar:<30}] {w:.3f}{marker}")

# HOW TO VERIFY:
#   - All values are between 0 and 1
#   - Each row sums to 1.0
#   - Higher weight = more attention to that token


# ============================================================
# PART 5: Weighted Sum — The Context-Aware Output
# ============================================================
# THE FINAL STEP:
#   output[i] = sum(attention_weight[i][j] * V[j]) for all j
#
#   Each token's output is a weighted average of ALL Value vectors,
#   where the weights come from attention.
#
# WHAT THIS MEANS:
#   If "bank" attends strongly to "river" (weight=0.4) and weakly
#   to "money" (weight=0.05), then bank's output will be heavily
#   influenced by river's Value vector, pulling its representation
#   toward "riverbank" meaning.

print("\n" + "=" * 60)
print("PART 5: Weighted Sum (Attention × Values)")
print("=" * 60)

# (1, seq_len, seq_len) × (1, seq_len, d_k) → (1, seq_len, d_k)
attention_output = torch.matmul(attention_weights, V)

print(f"\nAttention weights: {attention_weights.shape}")
print(f"Value vectors:     {V.shape}")
print(f"Output:            {attention_output.shape}")

# Show that output is different from input
input_vec = embeddings_output[0, 1, :5].detach()  # "the" token input
output_vec = attention_output[0, 1, :5].detach()   # "the" token after attention

print(f"\n'the' token:")
print(f"  Before attention (first 5 dims): {[f'{v:.4f}' for v in input_vec.tolist()]}")
print(f"  After attention (first 5 dims):  {[f'{v:.4f}' for v in output_vec.tolist()]}")
print(f"  → Different! The token now carries context from other tokens.")

# Verify: manually compute for one token
manual_output = torch.zeros(d_k)
for j in range(seq_len):
    manual_output += attention_weights[0, 1, j] * V[0, j]

print(f"\n✓ Manual matches matmul: {torch.allclose(manual_output, attention_output[0, 1], atol=1e-5)}")


# ============================================================
# PART 6: Complete Self-Attention in One Function
# ============================================================
# Putting it all together: the complete attention formula
#   Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

print("\n" + "=" * 60)
print("PART 6: Complete Self-Attention Function")
print("=" * 60)

def self_attention(embeddings, W_Q, W_K, W_V):
    """
    The complete self-attention mechanism.
    Input:  embeddings (batch, seq_len, d_model)
    Output: context-aware representations (batch, seq_len, d_k)
    """
    Q = W_Q(embeddings)
    K = W_K(embeddings)
    V = W_V(embeddings)

    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)

    return output, weights

with torch.no_grad():
    output, weights = self_attention(embeddings_output, W_Q, W_K, W_V)

print(f"\nInput:  {embeddings_output.shape}  (batch, seq_len, {d_model})")
print(f"Output: {output.shape}  (batch, seq_len, {d_k})")
print(f"\nThat's it. 6 lines of code. This IS the core of the transformer.")

# Verify it matches our step-by-step computation
print(f"✓ Matches step-by-step: {torch.allclose(output, attention_output, atol=1e-5)}")


# ============================================================
# PART 7: Multi-Head Attention — Why Multiple Heads?
# ============================================================
# WHY MULTIPLE HEADS?
#   One attention head learns ONE type of relationship:
#     Head 1 might learn: subject-verb relationships
#     Head 2 might learn: adjective-noun relationships
#     Head 3 might learn: coreference (pronoun → noun it refers to)
#
#   BERT-base has 12 heads, each with d_k = 64.
#   12 heads × 64 dims = 768 (back to original dimension).
#
# HOW IT WORKS:
#   1. Run self-attention 12 times in parallel (different W_Q, W_K, W_V each time)
#   2. Concatenate all 12 outputs: 12 × 64 = 768
#   3. One final linear projection to mix them together

print("\n" + "=" * 60)
print("PART 7: Multi-Head Attention From Scratch")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V (all at once, then split into heads)
        Q = self.W_Q(x)  # (batch, seq, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape: (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention for ALL heads at once (batched matrix multiply)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)

        # Concatenate heads: (batch, num_heads, seq, d_k) → (batch, seq, d_model)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final projection to mix head outputs
        output = self.W_O(attended)

        return output, weights

mha = MultiHeadAttention(d_model=768, num_heads=12)

with torch.no_grad():
    mha_output, mha_weights = mha(embeddings_output)

print(f"\nMulti-Head Attention:")
print(f"  Input:          {embeddings_output.shape}  (batch, seq, 768)")
print(f"  Output:         {mha_output.shape}  (batch, seq, 768) — same shape!")
print(f"  Weights shape:  {mha_weights.shape}  (batch, 12 heads, seq, seq)")

print(f"\n  Each head has its own {seq_len}×{seq_len} attention pattern.")
print(f"  12 heads × 64 dims = 768 dims (back to original size)")

# Show that different heads attend differently
print(f"\nDifferent heads, different attention patterns for 'cat' (token 2):")
for head_idx in [0, 3, 7, 11]:
    head_weights = mha_weights[0, head_idx, 2].detach().tolist()
    top_token = tokens[head_weights.index(max(head_weights))]
    print(f"  Head {head_idx:>2}: {[f'{w:.3f}' for w in head_weights]}  → attends most to '{top_token}'")

# HOW TO VERIFY:
#   - Output shape = input shape (768-dim in, 768-dim out)
#   - 12 heads × 64 = 768
#   - Different heads have different attention patterns
#   - Each head's attention row sums to 1.0


# ============================================================
# PART 8: Compare With BERT's Actual Attention
# ============================================================
# Let's see what BERT's trained attention actually looks like
# vs our random (untrained) attention.

print("\n" + "=" * 60)
print("PART 8: BERT's Trained Attention vs Our Random Attention")
print("=" * 60)

with torch.no_grad():
    bert_output = model(**encoded, output_attentions=True)

# BERT returns attention from all 12 layers, each with 12 heads
print(f"\nBERT has {len(bert_output.attentions)} layers of attention")
print(f"Each layer: {bert_output.attentions[0].shape}  (batch, 12 heads, seq, seq)")

# Show Layer 1, Head 0's attention pattern
layer = 0
head = 0
bert_attn = bert_output.attentions[layer][0, head]

print(f"\nBERT Layer {layer+1}, Head {head+1} attention pattern:")
print(f"{'':>10}", end="")
for t in tokens:
    print(f"{t:>10}", end="")
print()

for i, token in enumerate(tokens):
    print(f"{token:>10}", end="")
    for j in range(seq_len):
        w = bert_attn[i, j].item()
        print(f"{w:>10.4f}", end="")
    print()

# What does [CLS] attend to in different layers?
print(f"\nWhat [CLS] attends to across layers (trained BERT):")
for layer_idx in [0, 5, 11]:
    attn = bert_output.attentions[layer_idx][0, 0, 0]  # layer, batch, head 0, [CLS] row
    top_idx = attn.argmax().item()
    print(f"  Layer {layer_idx+1:>2}: {[f'{w:.3f}' for w in attn.tolist()]}  → most: '{tokens[top_idx]}'")


# ============================================================
# PART 9: The "bank" Example — See Context in Action
# ============================================================
# This is where it all comes together.
# Same word, different context → different attention patterns.

print("\n" + "=" * 60)
print("PART 9: 'bank' in Different Contexts — Attention Reveals Meaning")
print("=" * 60)

sentences = [
    "I went to the river bank to fish",
    "I went to the bank to deposit money",
]

for text in sentences:
    enc = tokenizer(text, return_tensors="pt")
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    bank_pos = toks.index("bank")

    with torch.no_grad():
        out = model(**enc, output_attentions=True)

    print(f"\n  Text: '{text}'")
    print(f"  Tokens: {toks}")
    print(f"  'bank' is at position {bank_pos}\n")

    # Show which tokens "bank" attends to in key layers
    for layer_idx in [0, 5, 11]:
        # Average across all 12 heads for cleaner signal
        attn = out.attentions[layer_idx][0, :, bank_pos, :].mean(dim=0)
        top3_indices = attn.topk(3).indices.tolist()
        top3_weights = attn.topk(3).values.tolist()

        top3_info = [(toks[idx], f"{w:.3f}") for idx, w in zip(top3_indices, top3_weights)]
        print(f"  Layer {layer_idx+1:>2} — 'bank' attends most to: {top3_info}")

print(f"""
  KEY INSIGHT:
  In "river bank to fish" — 'bank' attends to 'river', 'fish'
  In "bank to deposit money" — 'bank' attends to 'deposit', 'money'

  The attention mechanism lets the model USE context words
  to disambiguate meaning. This is self-attention in action.
""")


# ============================================================
# PART 10: Masked Attention (for GPT-style models)
# ============================================================
# BERT: bidirectional — each token sees ALL other tokens
# GPT:  causal/masked — each token only sees tokens BEFORE it
#
# WHY?
#   GPT generates text left to right. When predicting the next word,
#   it can't look at future words (that would be cheating).
#   We enforce this with a MASK that blocks future positions.

print("=" * 60)
print("PART 10: Masked (Causal) Attention")
print("=" * 60)

def masked_self_attention(embeddings, W_Q, W_K, W_V):
    Q = W_Q(embeddings)
    K = W_K(embeddings)
    V = W_V(embeddings)

    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Create causal mask: upper triangle = -inf (can't attend to future)
    seq_len = scores.shape[-1]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)

    return output, weights

with torch.no_grad():
    masked_out, masked_weights = masked_self_attention(embeddings_output, W_Q, W_K, W_V)

print(f"\nMasked attention weights (lower triangle only):")
print(f"{'':>10}", end="")
for t in tokens:
    print(f"{t:>10}", end="")
print()

for i, token in enumerate(tokens):
    print(f"{token:>10}", end="")
    for j in range(seq_len):
        w = masked_weights[0, i, j].item()
        print(f"{w:>10.4f}", end="")
    print()

print(f"""
  Notice:
  - Row 1 ([CLS]):  can only see itself → weight 1.000
  - Row 2 ('the'):   can see [CLS] and itself → weights split between two
  - Row 3 ('cat'):   can see [CLS], 'the', itself → weights split three ways
  - Last row:        can see everything → same as regular attention

  Upper triangle is all 0.000 — future tokens are invisible.
  BERT uses the unmasked version (Part 4). GPT uses this masked version.
""")

# HOW TO VERIFY:
#   - Upper triangle of weights = 0 (no attending to future)
#   - Each row still sums to 1.0
#   - Row i has exactly (i+1) non-zero values

for i in range(seq_len):
    row = masked_weights[0, i]
    non_zero = (row > 1e-6).sum().item()
    row_sum = row.sum().item()
    print(f"  Row {i} ({tokens[i]:>10}): {int(non_zero)} non-zero values, sum = {row_sum:.4f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: Self-Attention Demystified")
print("=" * 60)
print("""
THE FORMULA:
  Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

STEP BY STEP:
  1. Q = x · W_Q    (what am I looking for?)
  2. K = x · W_K    (what do I offer?)
  3. V = x · W_V    (what info do I carry?)
  4. scores = Q · K^T / sqrt(d_k)   (compatibility scores)
  5. weights = softmax(scores)       (normalize to probabilities)
  6. output = weights · V            (weighted mix of values)

MULTI-HEAD:
  Run this 12 times with different W_Q, W_K, W_V → concatenate → project
  Each head captures a different type of relationship

MASKED vs UNMASKED:
  BERT  = unmasked (sees all tokens) — used for understanding
  GPT   = masked (sees only past tokens) — used for generation

THE FLOW SO FAR:
  Text → Tokenizer → IDs → Embeddings → Self-Attention → Context-Aware Vectors
       (Step 1)          (Step 2)      (Step 3 - you are here)

NEXT STEP: 04_transformer_block.py — the full transformer layer
           (self-attention + feed-forward + residual connections + layer norm)
           Then you'll have the complete picture.
""")

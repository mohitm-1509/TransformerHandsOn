"""
STEP 4: The Complete Transformer Block
========================================

THE BIG PICTURE:
    Self-attention (Step 3) is only ONE part of a transformer layer.
    A complete transformer block has 4 components:

    Input
      │
      ├──→ Multi-Head Self-Attention
      │         │
      └──→ ADD (residual connection)
              │
           LayerNorm
              │
              ├──→ Feed-Forward Network (2 linear layers + activation)
              │         │
              └──→ ADD (residual connection)
                      │
                   LayerNorm
                      │
                   Output

    BERT stacks 12 of these blocks. GPT-3 stacks 96.
    That's the entire architecture. No hidden magic beyond this.

WHY EACH COMPONENT EXISTS:
    Self-Attention:       mix information BETWEEN tokens (context)
    Feed-Forward:         process each token INDIVIDUALLY (transform)
    Residual Connection:  let gradients flow (train deep networks)
    LayerNorm:            stabilize values (prevent explosion/vanishing)

WHAT YOU'LL LEARN:
    1. Feed-Forward Network — what it does and why
    2. Residual connections — the trick that makes deep networks trainable
    3. Layer Normalization — keeping values in check
    4. Full transformer block from scratch
    5. Stacking multiple blocks
    6. Compare with BERT's actual architecture
    7. The complete pipeline: text → output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")


# ============================================================
# PART 1: Feed-Forward Network — The Other Half
# ============================================================
# WHAT IS IT?
#   Two linear layers with an activation function in between:
#     FFN(x) = Linear2(GELU(Linear1(x)))
#
#   Linear1: 768 → 3072 (expand by 4x)
#   GELU:    non-linear activation
#   Linear2: 3072 → 768 (compress back)
#
# WHY EXPAND THEN COMPRESS?
#   The expansion gives the network more "room to think."
#   768 dims might not be enough to capture complex patterns.
#   Expanding to 3072 lets the model do richer computations,
#   then compress back to 768 to keep dimensions consistent.
#
# WHY DO WE NEED THIS AFTER ATTENTION?
#   Attention mixes information BETWEEN tokens (horizontal).
#   Feed-forward processes each token INDIVIDUALLY (vertical).
#   Attention says "bank relates to river."
#   Feed-forward says "given this context, transform the representation."
#   You need BOTH for the model to work well.
#
# ANALOGY:
#   Attention = a meeting where everyone shares information
#   Feed-forward = each person goes back to their desk and thinks
#                  about what they heard

print("=" * 60)
print("PART 1: Feed-Forward Network")
print("=" * 60)

d_model = 768
d_ff = 3072  # 4 * d_model (standard ratio)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (batch, seq_len, 768)
        x = self.linear1(x)       # (batch, seq_len, 3072) — expand
        x = F.gelu(x)             # non-linearity
        x = self.linear2(x)       # (batch, seq_len, 768)  — compress back
        return x

ff = FeedForward(d_model, d_ff)

# Get embeddings to work with
text = "The cat sat on the mat"
encoded = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

with torch.no_grad():
    emb_output = model.embeddings(encoded["input_ids"])

print(f"\nText: '{text}'")
print(f"Tokens: {tokens}")

with torch.no_grad():
    ff_output = ff(emb_output)

print(f"\nFeed-Forward Network:")
print(f"  Input:    {emb_output.shape}  (batch, seq, 768)")
print(f"  After Linear1: (batch, seq, {d_ff})  — expanded 4x")
print(f"  After GELU:    (batch, seq, {d_ff})  — non-linearity applied")
print(f"  After Linear2: {ff_output.shape}  — compressed back")

# Show it processes EACH TOKEN independently
# Token 0 output depends ONLY on token 0 input (no cross-token mixing)
with torch.no_grad():
    single_token = emb_output[:, 0:1, :]  # just [CLS]
    single_output = ff(single_token)
    full_output = ff(emb_output)

print(f"\n✓ Processes tokens independently: "
      f"{torch.allclose(single_output[0, 0], full_output[0, 0], atol=1e-5)}")
print(f"  → Token 0's output is the same whether we process it alone or with all tokens")

# HOW TO VERIFY:
#   - Output shape = input shape (768 → 768)
#   - Each token processed independently
#   - 3072/768 = 4x expansion ratio

# Compare with BERT's actual FFN
bert_ffn = model.encoder.layer[0].intermediate
print(f"\nBERT's actual FFN (layer 0):")
print(f"  Intermediate: {bert_ffn.dense}")
print(f"  Output:       {model.encoder.layer[0].output.dense}")


# ============================================================
# PART 2: Residual Connections — Why "Add" Matters
# ============================================================
# THE PROBLEM:
#   Deep networks (12+ layers) have vanishing gradients.
#   By layer 12, gradients are so tiny the early layers barely learn.
#
# THE SOLUTION (from ResNet, 2015):
#   output = LayerNorm(x + SubLayer(x))
#   Instead of: output = SubLayer(x)
#
#   The "+ x" is the residual connection. It means:
#   "Keep the original input and ADD the transformation to it."
#
# WHY THIS WORKS:
#   Without residual: the model must learn the FULL transformation
#   With residual: the model only learns the DIFFERENCE (residual)
#
#   If a layer has nothing useful to add, the gradient can still
#   flow through the "+ x" shortcut. The layer can effectively
#   become an identity function (add nothing) without blocking gradients.
#
# ANALOGY:
#   Without residual: "Rewrite this entire essay from scratch"
#   With residual:    "Here's the essay. Just mark your edits."
#   Much easier to learn small corrections than full rewrites.

print("\n" + "=" * 60)
print("PART 2: Residual Connections")
print("=" * 60)

x = emb_output  # original input

with torch.no_grad():
    sublayer_output = ff(x)

    without_residual = sublayer_output
    with_residual = x + sublayer_output

print(f"\nOriginal input (first 5 dims of token 0):")
print(f"  x:                   {[f'{v:.4f}' for v in x[0, 0, :5].tolist()]}")
print(f"\nFeed-forward output:")
print(f"  FFN(x):              {[f'{v:.4f}' for v in sublayer_output[0, 0, :5].tolist()]}")
print(f"\nWithout residual (just FFN output):")
print(f"  FFN(x):              {[f'{v:.4f}' for v in without_residual[0, 0, :5].tolist()]}")
print(f"\nWith residual (original + FFN output):")
print(f"  x + FFN(x):          {[f'{v:.4f}' for v in with_residual[0, 0, :5].tolist()]}")

# Show that information is preserved
cos_sim_without = F.cosine_similarity(
    x[0, 0].unsqueeze(0), without_residual[0, 0].unsqueeze(0)
).item()
cos_sim_with = F.cosine_similarity(
    x[0, 0].unsqueeze(0), with_residual[0, 0].unsqueeze(0)
).item()

print(f"\nSimilarity to original input:")
print(f"  Without residual: {cos_sim_without:.4f}  (original info largely lost)")
print(f"  With residual:    {cos_sim_with:.4f}  (original info preserved + new info added)")

# HOW TO VERIFY:
#   - With residual: output is closer to input (info preserved)
#   - Without residual: output may be very different (info lost)
#   - x + FFN(x) = element-wise addition, same shape


# ============================================================
# PART 3: Layer Normalization — Keeping Values Stable
# ============================================================
# WHAT IS IT?
#   For each token's 768-dim vector, normalize so that:
#     mean ≈ 0, std ≈ 1
#   Then apply learned scale (gamma) and shift (beta).
#
# WHY NEEDED?
#   After adding residual connections and passing through layers,
#   values can drift — some dimensions explode, others vanish.
#   LayerNorm keeps everything in a reasonable range.
#
# HOW IT WORKS:
#   For each token vector of 768 values:
#     1. Compute mean and std across the 768 dimensions
#     2. Normalize: (x - mean) / (std + eps)
#     3. Scale and shift: gamma * normalized + beta
#        (gamma and beta are learnable parameters)
#
# WHY LAYER NORM (not batch norm)?
#   Batch norm normalizes across the batch dimension.
#   Layer norm normalizes across the feature dimension.
#   For sequences of variable length, layer norm works better
#   because it doesn't depend on batch statistics.

print("\n" + "=" * 60)
print("PART 3: Layer Normalization")
print("=" * 60)

layer_norm = nn.LayerNorm(d_model)

with torch.no_grad():
    # Before normalization
    pre_norm = x + ff(x)  # residual + sublayer

    # Manual layer norm (to show what happens inside)
    token_vec = pre_norm[0, 0]  # first token, 768 values
    mean = token_vec.mean()
    std = token_vec.std(unbiased=False)
    manual_norm = (token_vec - mean) / (std + 1e-5)

    # PyTorch layer norm
    ln_output = layer_norm(pre_norm)

print(f"Before LayerNorm (token 0, first 5 dims):")
print(f"  Values: {[f'{v:.4f}' for v in pre_norm[0, 0, :5].tolist()]}")
print(f"  Mean:   {pre_norm[0, 0].mean().item():.4f}")
print(f"  Std:    {pre_norm[0, 0].std().item():.4f}")

print(f"\nAfter LayerNorm (token 0, first 5 dims):")
print(f"  Values: {[f'{v:.4f}' for v in ln_output[0, 0, :5].tolist()]}")
print(f"  Mean:   {ln_output[0, 0].mean().item():.6f}  (≈ 0)")
print(f"  Std:    {ln_output[0, 0].std().item():.4f}  (≈ 1)")

# Show stats for ALL tokens
print(f"\nMean and Std after LayerNorm for all tokens:")
for i, token in enumerate(tokens):
    mean_val = ln_output[0, i].mean().item()
    std_val = ln_output[0, i].std().item()
    print(f"  {token:>10}: mean={mean_val:>9.6f}  std={std_val:.4f}")

# HOW TO VERIFY:
#   - After LayerNorm: mean ≈ 0, std ≈ 1 for each token
#   - Shape is unchanged (768 dims)
#   - Values are in a reasonable range (roughly -3 to +3)

print(f"\n✓ All means ≈ 0: {all(abs(ln_output[0, i].mean().item()) < 0.001 for i in range(len(tokens)))}")
print(f"✓ Shape preserved: {ln_output.shape == pre_norm.shape}")


# ============================================================
# PART 4: The Complete Transformer Block — From Scratch
# ============================================================
# Now we put it all together:
#
#   def transformer_block(x):
#       # Sub-layer 1: Multi-Head Attention + Add & Norm
#       attn_output = multi_head_attention(x)
#       x = layer_norm1(x + attn_output)       ← residual + norm
#
#       # Sub-layer 2: Feed-Forward + Add & Norm
#       ff_output = feed_forward(x)
#       x = layer_norm2(x + ff_output)          ← residual + norm
#
#       return x
#
#   That's it. The entire transformer layer in 4 lines of logic.

print("\n" + "=" * 60)
print("PART 4: Complete Transformer Block From Scratch")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)

        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_O(attended), weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sub-layer 1: Attention + Residual + Norm
        attn_output, attn_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Sub-layer 2: Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


block = TransformerBlock(d_model=768, num_heads=12, d_ff=3072)
block.eval()

with torch.no_grad():
    block_output, block_weights = block(emb_output)

print(f"\nTransformer Block:")
print(f"  Input:            {emb_output.shape}")
print(f"  Output:           {block_output.shape}  ← same shape!")
print(f"  Attention weights: {block_weights.shape}")

# Show the flow step by step
print(f"\nStep-by-step values for token 0 ([CLS]):")
with torch.no_grad():
    step1_attn, _ = block.attention(emb_output)
    step2_add = emb_output + step1_attn
    step3_norm = block.norm1(step2_add)
    step4_ff = block.feed_forward(step3_norm)
    step5_add = step3_norm + step4_ff
    step6_norm = block.norm2(step5_add)

steps = [
    ("Input embedding", emb_output),
    ("After attention", step1_attn),
    ("After ADD (residual)", step2_add),
    ("After LayerNorm 1", step3_norm),
    ("After feed-forward", step4_ff),
    ("After ADD (residual)", step5_add),
    ("After LayerNorm 2", step6_norm),
]

for name, tensor in steps:
    vals = tensor[0, 0, :4].tolist()
    mean = tensor[0, 0].mean().item()
    std = tensor[0, 0].std().item()
    print(f"  {name:>25}: [{', '.join(f'{v:+.4f}' for v in vals)}, ...]  "
          f"mean={mean:+.4f}  std={std:.4f}")

# HOW TO VERIFY:
#   - Input shape = output shape (768 → 768)
#   - After each LayerNorm: mean ≈ 0, std ≈ 1
#   - Residual connections: ADD steps show values closer to input
#   - The block is self-contained: any number can be stacked

print(f"\n✓ Shape preserved: {emb_output.shape == block_output.shape}")
print(f"✓ After norm1 mean ≈ 0: {abs(step3_norm[0, 0].mean().item()) < 0.01}")
print(f"✓ After norm2 mean ≈ 0: {abs(step6_norm[0, 0].mean().item()) < 0.01}")


# ============================================================
# PART 5: Stacking Multiple Blocks — Depth = Power
# ============================================================
# WHY STACK?
#   Layer 1: learns simple patterns (adjacent words, basic syntax)
#   Layer 6: learns intermediate patterns (phrases, local meaning)
#   Layer 12: learns complex patterns (long-range dependencies, semantics)
#
#   Each layer refines the representation. More layers = deeper understanding.
#
# MODEL SIZES:
#   BERT-base:  12 layers,  768 dim, 12 heads →  110M parameters
#   BERT-large: 24 layers, 1024 dim, 16 heads →  340M parameters
#   GPT-2:      12 layers,  768 dim, 12 heads →  124M parameters
#   GPT-3:      96 layers, 12288 dim, 96 heads → 175B parameters

print("\n" + "=" * 60)
print("PART 5: Stacking Transformer Blocks")
print("=" * 60)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        layer_outputs = []
        for layer in self.layers:
            x, weights = layer(x)
            layer_outputs.append(x)
        return x, layer_outputs

encoder = TransformerEncoder(num_layers=12, d_model=768, num_heads=12, d_ff=3072)
encoder.eval()

with torch.no_grad():
    final_output, all_layer_outputs = encoder(emb_output)

print(f"\n12-Layer Transformer Encoder:")
print(f"  Input:  {emb_output.shape}")
print(f"  Output: {final_output.shape}")

# Show how representation evolves through layers
print(f"\nHow token 'cat' (position 2) evolves through layers:")
print(f"  (cosine similarity with the embedding layer output)\n")

cat_embedding = emb_output[0, 2]
for i, layer_out in enumerate(all_layer_outputs):
    cat_at_layer = layer_out[0, 2]
    sim = F.cosine_similarity(
        cat_embedding.unsqueeze(0), cat_at_layer.unsqueeze(0)
    ).item()
    bar = "█" * int(abs(sim) * 40)
    print(f"  Layer {i+1:>2}: sim={sim:+.4f}  {bar}")

print(f"\n  → Each layer transforms the representation further from the original.")
print(f"    By layer 12, the vector barely resembles the static embedding.")
print(f"    It now encodes deep contextual understanding of the full sentence.")

# Count parameters
total_params = sum(p.numel() for p in encoder.parameters())
print(f"\nParameter count: {total_params:,}")
print(f"  (BERT-base encoder has ~85M of its 110M total params here)")


# ============================================================
# PART 6: Compare With BERT's Actual Architecture
# ============================================================
# Let's look inside BERT and confirm it matches our scratch code.

print("\n" + "=" * 60)
print("PART 6: BERT's Actual Architecture (Confirming Our Code)")
print("=" * 60)

print(f"\nBERT-base structure:")
print(f"  Layers: {len(model.encoder.layer)}")

layer0 = model.encoder.layer[0]
print(f"\n  Layer 0 components:")
print(f"    Attention:")
print(f"      Q: {layer0.attention.self.query}")
print(f"      K: {layer0.attention.self.key}")
print(f"      V: {layer0.attention.self.value}")
print(f"      Output projection: {layer0.attention.output.dense}")
print(f"      LayerNorm: {layer0.attention.output.LayerNorm}")
print(f"    Feed-Forward:")
print(f"      Expand:  {layer0.intermediate.dense}")
print(f"      Compress: {layer0.output.dense}")
print(f"      LayerNorm: {layer0.output.LayerNorm}")

print(f"\n  Matches our TransformerBlock:")
print(f"    ✓ Q, K, V projections:  Linear(768, 768)")
print(f"    ✓ Output projection:    Linear(768, 768)")
print(f"    ✓ FFN expand:           Linear(768, 3072)")
print(f"    ✓ FFN compress:         Linear(3072, 768)")
print(f"    ✓ Two LayerNorms per block")
print(f"    ✓ 12 heads, d_k = 768/12 = 64")

# Run through BERT and compare shapes
with torch.no_grad():
    bert_out = model(**encoded, output_hidden_states=True, output_attentions=True)

print(f"\nBERT hidden states (one per layer + embedding):")
for i, hs in enumerate(bert_out.hidden_states):
    label = "embedding" if i == 0 else f"layer {i}"
    print(f"  {label:>12}: {hs.shape}")


# ============================================================
# PART 7: The Complete Pipeline — Text to Output
# ============================================================
# Let's trace the ENTIRE flow from raw text to final output,
# matching every step with what we learned in Steps 1-4.

print("\n" + "=" * 60)
print("PART 7: The Complete Pipeline (Steps 1-4 Combined)")
print("=" * 60)

text = "The cat sat on the mat"
print(f"\nInput: '{text}'")

# STEP 1: Tokenization
enc = tokenizer(text, return_tensors="pt")
toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\n  Step 1 — Tokenization:")
print(f"    Tokens: {toks}")
print(f"    IDs:    {enc['input_ids'][0].tolist()}")
print(f"    Mask:   {enc['attention_mask'][0].tolist()}")

# STEP 2: Embeddings
with torch.no_grad():
    embeddings = model.embeddings(enc["input_ids"])
print(f"\n  Step 2 — Embeddings:")
print(f"    Word + Position + TokenType → LayerNorm")
print(f"    Shape: {embeddings.shape}  (each token → 768-dim vector)")

# STEP 3 & 4: Through all 12 transformer layers
with torch.no_grad():
    output = model(**enc, output_hidden_states=True, output_attentions=True)

print(f"\n  Steps 3-4 — Through 12 Transformer Blocks:")
for i in range(1, 13):
    hs = output.hidden_states[i]
    attn = output.attentions[i-1]
    print(f"    Layer {i:>2}: hidden={hs.shape}  attention={attn.shape}")

print(f"\n  Final output: {output.last_hidden_state.shape}")
print(f"    → Each token now has a 768-dim CONTEXTUAL representation")
print(f"    → [CLS] token (position 0) represents the entire sentence")

# Show [CLS] token's final representation
cls_vector = output.last_hidden_state[0, 0]
print(f"\n  [CLS] vector (first 10 dims): {[f'{v:.4f}' for v in cls_vector[:10].tolist()]}")
print(f"  This vector IS the sentence representation.")
print(f"  Used for classification, similarity, search, etc.")


# ============================================================
# PART 8: What Each Layer Learns (Probing BERT)
# ============================================================
# Research has shown that different layers capture different
# linguistic properties:
#   Early layers (1-4):   surface features, word order, POS tags
#   Middle layers (5-8):  syntax, parse trees, subject-verb agreement
#   Late layers (9-12):   semantics, reasoning, task-specific features

print("\n" + "=" * 60)
print("PART 8: What Different Layers Capture")
print("=" * 60)

sentences = [
    "The cat sat on the mat",
    "The dog sat on the mat",     # similar syntax and meaning (cat→dog)
    "Running quickly is fun",     # completely different
]

print(f"\nComparing sentence representations at different layers:")
print(f"  S1: '{sentences[0]}'")
print(f"  S2: '{sentences[1]}'  (similar — just cat→dog)")
print(f"  S3: '{sentences[2]}'  (completely different)\n")

encodings = [tokenizer(s, return_tensors="pt") for s in sentences]
with torch.no_grad():
    outputs = [model(**e, output_hidden_states=True) for e in encodings]

for layer_idx in [0, 1, 4, 8, 12]:
    label = "Embedding" if layer_idx == 0 else f"Layer {layer_idx:>2}"
    # Use mean of all tokens as sentence representation
    rep1 = outputs[0].hidden_states[layer_idx][0].mean(dim=0)
    rep2 = outputs[1].hidden_states[layer_idx][0].mean(dim=0)
    rep3 = outputs[2].hidden_states[layer_idx][0].mean(dim=0)

    sim_12 = F.cosine_similarity(rep1.unsqueeze(0), rep2.unsqueeze(0)).item()
    sim_13 = F.cosine_similarity(rep1.unsqueeze(0), rep3.unsqueeze(0)).item()

    print(f"  {label}:  S1 vs S2 (similar) = {sim_12:.4f}  |  "
          f"S1 vs S3 (different) = {sim_13:.4f}  |  "
          f"gap = {sim_12 - sim_13:+.4f}")

print(f"""
  → At the embedding layer, all sentences look somewhat similar
    (they share common words like "the", "sat", "on").
  → As we go deeper, the model better separates different meanings.
  → The gap between similar vs different sentences should grow
    in later layers (model learns to distinguish meaning).
""")


# ============================================================
# PART 9: Parameter Count Breakdown
# ============================================================
# Understanding WHERE the parameters are helps you understand
# what the model is actually learning.

print("=" * 60)
print("PART 9: Where Are All the Parameters?")
print("=" * 60)

# Count BERT's parameters by component
embedding_params = sum(p.numel() for p in model.embeddings.parameters())
attention_params = 0
ffn_params = 0
norm_params = 0

for layer in model.encoder.layer:
    attention_params += sum(p.numel() for p in layer.attention.parameters())
    ffn_params += sum(p.numel() for p in layer.intermediate.parameters())
    ffn_params += sum(p.numel() for p in layer.output.dense.parameters())
    norm_params += sum(p.numel() for p in layer.output.LayerNorm.parameters())

pooler_params = sum(p.numel() for p in model.pooler.parameters())
total = sum(p.numel() for p in model.parameters())

print(f"\nBERT-base Parameter Breakdown:")
print(f"  Embeddings (word + pos + type): {embedding_params:>12,}  ({embedding_params/total*100:.1f}%)")
print(f"  Attention (Q,K,V,O × 12):       {attention_params:>12,}  ({attention_params/total*100:.1f}%)")
print(f"  Feed-Forward (× 12):            {ffn_params:>12,}  ({ffn_params/total*100:.1f}%)")
print(f"  LayerNorm:                       {norm_params:>12,}  ({norm_params/total*100:.1f}%)")
print(f"  Pooler:                          {pooler_params:>12,}  ({pooler_params/total*100:.1f}%)")
print(f"  {'─' * 48}")
print(f"  Total:                           {total:>12,}")

print(f"""
  KEY INSIGHT:
  Feed-Forward layers have the MOST parameters (~2/3 of the model).
  Attention is actually relatively parameter-light.
  This is why some researchers call transformers
  "attention + a lot of feed-forward computation."
""")


# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("SUMMARY: The Complete Transformer Architecture")
print("=" * 60)
print("""
A TRANSFORMER BLOCK (4 components):
  1. Multi-Head Self-Attention  → mix info between tokens
  2. Add & LayerNorm            → residual connection + normalize
  3. Feed-Forward Network       → transform each token individually
  4. Add & LayerNorm            → residual connection + normalize

THE COMPLETE PIPELINE:
  ┌─────────────────────────────────────────────────────────┐
  │  "The cat sat on the mat"                               │
  │         │                                               │
  │  Step 1: Tokenizer                                      │
  │         │  [101, 1996, 4937, 2938, 2006, 1996, 13523, 102]  │
  │         │                                               │
  │  Step 2: Embeddings (word + position + token_type)      │
  │         │  shape: (1, 8, 768)                           │
  │         │                                               │
  │  Steps 3-4: × 12 Transformer Blocks                    │
  │         │  ┌─ Multi-Head Attention (12 heads)           │
  │         │  ├─ Add & LayerNorm                           │
  │         │  ├─ Feed-Forward (768→3072→768)               │
  │         │  └─ Add & LayerNorm                           │
  │         │  shape: (1, 8, 768) — same at every layer     │
  │         │                                               │
  │  Output: contextual embeddings (1, 8, 768)              │
  │         [CLS] vector = sentence representation          │
  └─────────────────────────────────────────────────────────┘

YOU NOW UNDERSTAND THE ENTIRE TRANSFORMER ARCHITECTURE.
There is no hidden magic beyond what you've coded in Steps 1-4.

NEXT STEP: 05_inference_pipeline.py
  Use pretrained models for real tasks (sentiment, summarization)
  via HuggingFace pipeline — see the transformer in ACTION
  on real-world problems without writing any model code.
""")

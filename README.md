# Transformer Hands-On: From Theory to Fine-Tuning

A progressive, code-first learning track that takes you from **"I understand transformers conceptually"** to **"I can confidently build NLP projects."**

Each script is self-contained, runnable, and includes inline reasoning explaining **what** each line does, **why** it exists, and **how to verify** the output is correct.

## The Problem This Solves

Most transformer tutorials either:
- Show math and diagrams (theory without code)
- Show HuggingFace one-liners (code without understanding)

This repo bridges the gap. You **code the internals from scratch**, then learn to **use the libraries** that wrap them — so when something breaks in a real project, you know exactly what's happening under the hood.

## The Learning Track

| Step | Script | What You Learn | Time |
|------|--------|---------------|------|
| 1 | `01_tokenization.py` | How text becomes numbers — subword splitting, input IDs, attention masks, padding, truncation, special tokens | 1 hr |
| 2 | `02_embeddings.py` | How IDs become vectors — word embeddings, positional embeddings, how they combine, semantic similarity proof | 1 hr |
| 3 | `03_self_attention.py` | The core mechanism — Q/K/V from scratch, attention scores, scaling, softmax, multi-head attention, masked (causal) attention | 2 hrs |
| 4 | `04_transformer_block.py` | The full architecture — feed-forward network, residual connections, LayerNorm, stacking layers, parameter breakdown | 2 hrs |
| 5 | `05_inference_pipeline.py` | Using pretrained models — sentiment analysis, NER, fill-mask, zero-shot classification, text generation, domain-specific models | 1 hr |
| 6 | `06_fine_tuning.py` | Training on your data — dataset loading, tokenization, Trainer API, evaluation metrics, saving/loading models | 3 hrs |

**Total: ~10 hours to mass the complete transformer pipeline.**

## The Flow

```
Text ──→ Tokenizer ──→ IDs ──→ Embeddings ──→ Self-Attention ──→ Transformer Blocks ──→ Output
         (Step 1)             (Step 2)       (Step 3)          (Step 4)

         Steps 1-4: Understanding the internals (code from scratch)
         Step 5:    Using pretrained models (one-line inference)
         Step 6:    Fine-tuning on your own data (real-world projects)
```

## Key Concepts Covered

- **Subword tokenization** — why "unhappiness" becomes `['un', '##ha', '##pp', '##iness']`
- **Word vs positional embeddings** — how the model knows both *what* a word means and *where* it sits
- **Self-attention from scratch** — the Q·K^T/√d_k · V formula, coded line by line
- **Why "bank" means different things** — contextual embeddings proven with cosine similarity
- **Multi-head attention** — why 12 heads each capture different linguistic patterns
- **Masked vs unmasked attention** — the difference between BERT and GPT
- **Residual connections** — the trick that makes deep networks trainable
- **LayerNorm** — keeping values stable across 12+ layers
- **Feed-forward networks** — the 768→3072→768 expand-compress pattern
- **Zero-shot classification** — classify text without any training data
- **Fine-tuning workflow** — the 6-step recipe for any text classification task

## Setup

```bash
pip install torch transformers datasets evaluate accelerate
```

Then run each script in order:

```bash
python 01_tokenization.py
python 02_embeddings.py
python 03_self_attention.py
python 04_transformer_block.py
python 05_inference_pipeline.py
python 06_fine_tuning.py
```

> **Note:** Step 6 (fine-tuning) takes ~10-15 minutes on CPU. On Mac with Apple Silicon, the script forces CPU to avoid MPS compatibility issues.

## Each Script Includes

- **Reasoning** — why each component exists and what problem it solves
- **Verification checks** — how to confirm the output is correct
- **Comparisons** — scratch code vs BERT's actual implementation
- **Visual output** — attention matrices, similarity bars, parameter breakdowns

## After This Track

You'll be ready to:
- Fine-tune BERT/FinBERT on domain-specific data (financial sentiment, medical NER, etc.)
- Understand what's happening when a model underperforms and how to debug it
- Read transformer research papers and map concepts to actual code
- Build production NLP systems with HuggingFace

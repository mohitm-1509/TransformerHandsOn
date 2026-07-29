"""
STEP 6: Fine-Tuning a Pretrained Model
========================================

THE BIG PICTURE:
    Step 5 showed you: use pretrained models as-is with pipeline.
    Step 6 shows you: TRAIN a pretrained model on YOUR data.

    Fine-tuning = take a model that already understands language (BERT)
                  and teach it YOUR specific task (e.g., financial sentiment).

WHY FINE-TUNE (not train from scratch)?
    Training BERT from scratch:
      - Needs ~16GB of text data
      - Needs 4-8 GPUs for 4+ days
      - Costs $10,000+ in compute
      - Result: a model that understands general language

    Fine-tuning BERT:
      - Needs 1,000-50,000 labeled examples
      - Needs 1 GPU (or even CPU) for 30 min - 2 hrs
      - Costs $1-10 in compute
      - Result: a model that understands YOUR specific task

    Fine-tuning is transfer learning:
      BERT already knows English grammar, word meanings, context.
      You just teach it "for MY task, positive means X, negative means Y."

WHAT HAPPENS DURING FINE-TUNING:
    1. Load pretrained BERT (all 110M params already trained)
    2. Add a classification head on top (a small linear layer)
    3. Feed YOUR labeled data through the model
    4. Adjust ALL weights slightly to minimize your task's loss
    5. The model adapts its language understanding to YOUR domain

WHAT YOU'LL LEARN:
    1. Load and explore a dataset
    2. Tokenize data for the model
    3. Set up model with classification head
    4. Configure training (learning rate, epochs, batch size)
    5. Train with HuggingFace Trainer
    6. Evaluate with real metrics
    7. Use your fine-tuned model for predictions
    8. Save and load the model
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "fine_tuned_model")

# ============================================================
# PART 1: Load and Explore the Dataset
# ============================================================
# WHAT DATASET?
#   "rotten_tomatoes" — movie review sentiment (positive/negative)
#   8,530 training examples, 1,066 test examples
#   Small enough to train on CPU in ~10-15 minutes.
#
# WHY THIS DATASET?
#   - Binary classification (simplest fine-tuning task)
#   - Small (fast to train, good for learning)
#   - Same domain as default sentiment model, so we can compare
#
# IN REAL PROJECTS:
#   You'd use YOUR OWN data here. For example:
#   - Your FinancialNewsSentiment data
#   - Customer support tickets with labels
#   - Medical notes with diagnoses

print("=" * 60)
print("PART 1: Loading the Dataset")
print("=" * 60)

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

print(f"\nDataset structure:")
print(f"  {dataset}")
print(f"\n  Training examples:   {len(dataset['train'])}")
print(f"  Validation examples: {len(dataset['validation'])}")
print(f"  Test examples:       {len(dataset['test'])}")

print(f"\n  Columns: {dataset['train'].column_names}")
print(f"  Labels:  0 = negative, 1 = positive")

# Look at some examples
print(f"\n  Sample data:")
for i in range(5):
    text = dataset["train"][i]["text"]
    label = dataset["train"][i]["label"]
    label_name = "positive" if label == 1 else "negative"
    print(f"    [{label_name:>8}] '{text[:80]}...'")

# Check label distribution
labels = dataset["train"]["label"]
pos_count = sum(labels)
neg_count = len(labels) - pos_count
print(f"\n  Label distribution:")
print(f"    Positive: {pos_count} ({pos_count/len(labels)*100:.1f}%)")
print(f"    Negative: {neg_count} ({neg_count/len(labels)*100:.1f}%)")

# HOW TO VERIFY:
#   - Dataset loaded without errors
#   - Has text and label columns
#   - Labels are balanced (roughly 50/50)
#   - Text looks like actual reviews


# ============================================================
# PART 2: Tokenize the Dataset
# ============================================================
# WHY TOKENIZE THE WHOLE DATASET UPFRONT?
#   The Trainer expects tokenized data, not raw text.
#   We tokenize once, cache it, and reuse during training.
#   This is faster than tokenizing on-the-fly each epoch.
#
# IMPORTANT CHOICES:
#   - max_length: how long each input can be (BERT max = 512)
#     Shorter = faster training, but may cut important text
#     128 is a good default for short texts (reviews, tweets)
#   - truncation=True: cut text longer than max_length
#   - padding="max_length": pad shorter texts to max_length
#     (some prefer dynamic padding for speed — we'll keep it simple)

print("\n" + "=" * 60)
print("PART 2: Tokenizing the Dataset")
print("=" * 60)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

print(f"\n  Tokenizing with '{model_name}' tokenizer...")
print(f"  Settings: max_length=128, padding=max_length, truncation=True")

start = time.time()
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(f"  Tokenized in {time.time() - start:.1f}s")

# Show what tokenization added
print(f"\n  Before tokenization columns: {dataset['train'].column_names}")
print(f"  After tokenization columns:  {tokenized_dataset['train'].column_names}")

# Inspect one tokenized example
example = tokenized_dataset["train"][0]
print(f"\n  Example (first training sample):")
print(f"    Text:    '{dataset['train'][0]['text'][:60]}...'")
print(f"    IDs:     {example['input_ids'][:20]}... (showing first 20 of {len(example['input_ids'])})")
print(f"    Mask:    {example['attention_mask'][:20]}... ")
print(f"    Label:   {example['label']}")

# Count padding
pad_count = example["input_ids"].count(0)
real_count = len(example["input_ids"]) - pad_count
print(f"    Real tokens: {real_count}, Padding tokens: {pad_count}")

# HOW TO VERIFY:
#   - New columns added: input_ids, attention_mask, (token_type_ids)
#   - All examples same length (128)
#   - input_ids start with 101 ([CLS]) and have 102 ([SEP]) somewhere
#   - attention_mask has 1s for real tokens, 0s for padding

print(f"\n✓ Starts with [CLS] (101): {example['input_ids'][0] == 101}")
print(f"✓ Contains [SEP] (102):    {102 in example['input_ids']}")
print(f"✓ Fixed length (128):      {len(example['input_ids']) == 128}")


# ============================================================
# PART 3: Load Model with Classification Head
# ============================================================
# WHAT IS AutoModelForSequenceClassification?
#   It's BERT + a classification head on top:
#
#   [CLS] token → 768-dim vector → Linear(768, num_labels) → logits
#
#   The Linear layer is the "classification head" — it's randomly
#   initialized (the ONLY new parameters). Everything else is pretrained.
#
# WHY num_labels=2?
#   We have 2 classes: positive and negative.
#   For 3-class sentiment (pos/neg/neutral), you'd use num_labels=3.
#   For NER, you'd use AutoModelForTokenClassification instead.
#
# WHAT GETS TRAINED?
#   During fine-tuning, ALL parameters are updated:
#   - The new classification head (randomly initialized) learns from scratch
#   - The BERT layers (pretrained) get SLIGHTLY adjusted
#   The learning rate is very small (2e-5) to avoid destroying
#   BERT's pretrained knowledge — this is key to fine-tuning.

print("\n" + "=" * 60)
print("PART 3: Loading Model with Classification Head")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
classifier_params = sum(p.numel() for p in model.classifier.parameters())

print(f"\n  Model: {model_name}")
print(f"  Task:  Sequence Classification (2 labels)")

print(f"\n  Parameter breakdown:")
print(f"    Total parameters:      {total_params:>12,}")
print(f"    Trainable parameters:  {trainable_params:>12,}  (100% — all fine-tuned)")
print(f"    Classifier head:       {classifier_params:>12,}  (the only NEW params)")
print(f"    Pretrained BERT:       {total_params - classifier_params:>12,}  (reused from pretraining)")

print(f"\n  Classifier head structure:")
print(f"    {model.classifier}")
print(f"    → Takes [CLS] token's 768-dim vector → outputs 2 logits")

# Show that the classifier is randomly initialized
with torch.no_grad():
    dummy_input = tokenizer("test sentence", return_tensors="pt")
    outputs = model(**dummy_input)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

print(f"\n  Before training (random classifier):")
print(f"    Logits: {logits[0].tolist()}")
print(f"    Probs:  {[f'{p:.4f}' for p in probs[0].tolist()]}")
print(f"    → Roughly 50/50 — classifier hasn't learned anything yet")

# HOW TO VERIFY:
#   - Classifier head is Linear(768, 2)
#   - Before training: predictions are ~50/50 (random)
#   - Total params ≈ 109M (BERT-base) + ~1,500 (classifier)


# ============================================================
# PART 4: Training Arguments — The Hyperparameters
# ============================================================
# WHAT ARE TRAINING ARGUMENTS?
#   Configuration for HOW to train — learning rate, batch size, etc.
#   Getting these right is crucial. Bad hyperparameters = bad model.
#
# KEY HYPERPARAMETERS EXPLAINED:
#
#   learning_rate = 2e-5 (0.00002)
#     Very small! We don't want to destroy BERT's pretrained knowledge.
#     Too high → model forgets what it learned in pretraining
#     Too low → model doesn't adapt to your task
#     2e-5 is the standard starting point for BERT fine-tuning.
#
#   num_train_epochs = 3
#     How many times the model sees the entire training set.
#     Fine-tuning usually needs only 2-4 epochs (not 100 like training from scratch).
#     More epochs → risk of overfitting on small datasets.
#
#   per_device_train_batch_size = 16
#     How many examples processed at once.
#     Larger = faster training, more memory
#     Smaller = less memory, potentially better generalization
#     16 is a good default for BERT on most GPUs. Use 8 if you run out of memory.
#
#   weight_decay = 0.01
#     Regularization to prevent overfitting.
#     Slightly penalizes large weights.

print("\n" + "=" * 60)
print("PART 4: Training Configuration")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=False,
    use_cpu=True,
)

print(f"\n  Training Configuration:")
print(f"    Learning rate:    {training_args.learning_rate}")
print(f"    Epochs:           {training_args.num_train_epochs}")
print(f"    Train batch size: {training_args.per_device_train_batch_size}")
print(f"    Eval batch size:  {training_args.per_device_eval_batch_size}")
print(f"    Weight decay:     {training_args.weight_decay}")
print(f"    FP16 (mixed):     {training_args.fp16}")
print(f"    Eval strategy:    {training_args.eval_strategy}")
print(f"    Device:           CPU (forced — MPS not fully supported for BERT)")

steps_per_epoch = len(tokenized_dataset["train"]) // training_args.per_device_train_batch_size
total_steps = steps_per_epoch * int(training_args.num_train_epochs)
print(f"\n  Estimated training:")
print(f"    Steps per epoch:  {steps_per_epoch}")
print(f"    Total steps:      {total_steps}")


# ============================================================
# PART 5: Evaluation Metrics
# ============================================================
# WHY NOT JUST USE LOSS?
#   Loss tells the model how to improve during training.
#   But humans understand accuracy, F1, precision, recall better.
#
# WHAT WE COMPUTE:
#   Accuracy:  % of predictions that are correct
#   F1 Score:  harmonic mean of precision and recall
#              (better than accuracy for imbalanced datasets)
#
# WHEN IS EACH USEFUL?
#   Balanced data (50/50 split):   accuracy is fine
#   Imbalanced (95/5 split):      F1 is essential
#     (A model that always says "negative" gets 95% accuracy
#      on a 95% negative dataset — but F1 would be 0 for positive class)

print("\n" + "=" * 60)
print("PART 5: Evaluation Metrics")
print("=" * 60)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {**accuracy, **f1}

# Demo with dummy data
dummy_preds = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
dummy_labels = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1])

acc = accuracy_metric.compute(predictions=dummy_preds, references=dummy_labels)
f1 = f1_metric.compute(predictions=dummy_preds, references=dummy_labels)

print(f"\n  Demo metrics (dummy data):")
print(f"    Predictions: {dummy_preds.tolist()}")
print(f"    Labels:      {dummy_labels.tolist()}")
print(f"    Accuracy:    {acc['accuracy']:.4f}")
print(f"    F1 Score:    {f1['f1']:.4f}")
correct = (dummy_preds == dummy_labels).sum()
print(f"    ({correct}/{len(dummy_labels)} correct predictions)")


# ============================================================
# PART 6: Train the Model!
# ============================================================
# WHAT THE TRAINER DOES:
#   1. Batches your data (groups examples together)
#   2. Forward pass (model predicts labels)
#   3. Compute loss (how wrong the predictions are)
#   4. Backward pass (compute gradients)
#   5. Update weights (optimizer step)
#   6. Repeat for all batches (1 epoch)
#   7. Evaluate on validation set
#   8. Repeat for all epochs
#   9. Save the best model
#
#   The Trainer does ALL of this. You just configure and call train().
#
# WHAT TO WATCH:
#   - Training loss should DECREASE over time
#   - Validation loss should DECREASE (if it increases → overfitting)
#   - Accuracy should INCREASE over epochs
#   - If val loss goes up while train loss goes down → stop early

print("\n" + "=" * 60)
print("PART 6: Training the Model")
print("=" * 60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Evaluate BEFORE training (baseline)
print(f"\n  Evaluating BEFORE training (random classifier)...")
pre_train_metrics = trainer.evaluate()
print(f"    Accuracy: {pre_train_metrics['eval_accuracy']:.4f}")
print(f"    F1:       {pre_train_metrics['eval_f1']:.4f}")
print(f"    Loss:     {pre_train_metrics['eval_loss']:.4f}")
print(f"    → Should be ~50% accuracy (random guessing)")

# Train!
print(f"\n  Starting training... (this takes ~10-15 min on CPU, ~2 min on GPU)")
print(f"  Watch the loss decrease and accuracy increase.\n")

start = time.time()
train_result = trainer.train()
train_time = time.time() - start

print(f"\n  Training completed in {train_time:.0f}s ({train_time/60:.1f} min)")
print(f"  Training loss:  {train_result.training_loss:.4f}")

# HOW TO VERIFY:
#   - Training loss should be much lower than starting loss
#   - Training should complete without errors
#   - Time should be reasonable (5-20 min CPU, 1-3 min GPU)


# ============================================================
# PART 7: Evaluate the Fine-Tuned Model
# ============================================================
# Now we test on HELD-OUT data the model never saw during training.
# This tells us how the model will perform on new, unseen text.

print("\n" + "=" * 60)
print("PART 7: Evaluating the Fine-Tuned Model")
print("=" * 60)

# Evaluate on validation set
val_metrics = trainer.evaluate()
print(f"\n  Validation Set Results:")
print(f"    Accuracy: {val_metrics['eval_accuracy']:.4f}")
print(f"    F1:       {val_metrics['eval_f1']:.4f}")
print(f"    Loss:     {val_metrics['eval_loss']:.4f}")

# Evaluate on test set
test_metrics = trainer.evaluate(tokenized_dataset["test"])
print(f"\n  Test Set Results:")
print(f"    Accuracy: {test_metrics['eval_accuracy']:.4f}")
print(f"    F1:       {test_metrics['eval_f1']:.4f}")
print(f"    Loss:     {test_metrics['eval_loss']:.4f}")

# Compare before and after
print(f"\n  Improvement:")
print(f"    Before training:  accuracy={pre_train_metrics['eval_accuracy']:.4f}")
print(f"    After training:   accuracy={val_metrics['eval_accuracy']:.4f}")
print(f"    Improvement:      +{(val_metrics['eval_accuracy'] - pre_train_metrics['eval_accuracy'])*100:.1f} percentage points")

# HOW TO VERIFY:
#   - Accuracy should be >80% (good fine-tuning on this dataset gets 85%+)
#   - Test accuracy close to validation accuracy (no overfitting)
#   - Huge improvement over the ~50% baseline


# ============================================================
# PART 8: Make Predictions with Your Fine-Tuned Model
# ============================================================
# Now use YOUR model to classify new text — the whole point of this exercise.

print("\n" + "=" * 60)
print("PART 8: Making Predictions")
print("=" * 60)

model.eval()

test_texts = [
    "This movie is absolutely fantastic, a true masterpiece!",
    "Terrible film. Complete waste of time and money.",
    "It was an okay movie, had some good moments.",
    "The acting was brilliant but the plot made no sense.",
    "One of the worst movies I have ever seen in my life.",
    "A delightful experience from start to finish.",
]

print(f"\n  Predictions with fine-tuned model:\n")
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    bar = "█" * int(confidence * 30)
    print(f"  [{label:>8}] ({confidence:.4f}) {bar}")
    print(f"    '{text}'\n")

# Compare with pipeline using the SAME fine-tuned model
from transformers import pipeline as hf_pipeline

fine_tuned_pipeline = hf_pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
)

print(f"  Same predictions via pipeline (easier API):")
results = fine_tuned_pipeline(test_texts)
for text, result in zip(test_texts, results):
    print(f"    [{result['label']:>8}] ({result['score']:.4f}) '{text[:50]}...'")

# HOW TO VERIFY:
#   - Clearly positive → POSITIVE with high confidence
#   - Clearly negative → NEGATIVE with high confidence
#   - Ambiguous → lower confidence
#   - Pipeline results should match manual predictions


# ============================================================
# PART 9: Save and Load Your Model
# ============================================================
# WHY SAVE?
#   You just spent 10+ minutes training. You don't want to retrain
#   every time you need the model. Save once, load instantly.
#
# WHAT GETS SAVED?
#   - Model weights (the trained parameters)
#   - Tokenizer (so you use the correct one)
#   - Config (model architecture details)
#
# HOW TO USE IN PRODUCTION?
#   Load the saved model → create a pipeline → serve predictions.
#   This is what your Flask/FastAPI app would do.

print("\n" + "=" * 60)
print("PART 9: Saving and Loading the Model")
print("=" * 60)

save_path = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\n  Model saved to: {save_path}")
print(f"  Files saved:")
for f in sorted(os.listdir(save_path)):
    size = os.path.getsize(os.path.join(save_path, f))
    size_mb = size / (1024 * 1024)
    if size_mb > 1:
        print(f"    {f:>40}  ({size_mb:.1f} MB)")
    else:
        print(f"    {f:>40}  ({size/1024:.1f} KB)")

# Load it back (simulating a fresh start)
print(f"\n  Loading model back from disk...")
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)

loaded_pipeline = hf_pipeline("sentiment-analysis", model=loaded_model, tokenizer=loaded_tokenizer)
test = loaded_pipeline("This movie was absolutely wonderful!")
print(f"  Loaded model prediction: {test[0]['label']} ({test[0]['score']:.4f})")
print(f"  ✓ Model loads and works correctly!")

# HOW TO VERIFY:
#   - Files exist in save_path
#   - Model file is ~440MB (BERT-base)
#   - Loaded model gives same predictions as the trained one


# ============================================================
# PART 10: The Complete Fine-Tuning Workflow (Summary)
# ============================================================

print("\n" + "=" * 60)
print("PART 10: The Complete Fine-Tuning Workflow")
print("=" * 60)
print(f"""
  THE RECIPE (same for ANY text classification task):

  ┌─────────────────────────────────────────────────────────┐
  │  1. DATASET                                             │
  │     dataset = load_dataset("your_data")                 │
  │                                                         │
  │  2. TOKENIZE                                            │
  │     tokenizer = AutoTokenizer.from_pretrained(model)    │
  │     tokenized = dataset.map(tokenize_fn, batched=True)  │
  │                                                         │
  │  3. MODEL                                               │
  │     model = AutoModelForSequenceClassification          │
  │              .from_pretrained(model, num_labels=N)      │
  │                                                         │
  │  4. TRAIN                                               │
  │     trainer = Trainer(model, args, data, metrics)       │
  │     trainer.train()                                     │
  │                                                         │
  │  5. EVALUATE                                            │
  │     trainer.evaluate(test_set)                          │
  │                                                         │
  │  6. SAVE & DEPLOY                                       │
  │     model.save_pretrained("./my_model")                 │
  │     pipeline("task", model="./my_model")                │
  └─────────────────────────────────────────────────────────┘

  YOUR NEXT STEPS — REAL PROJECTS:

  FinancialNewsSentiment:
    Same workflow, but:
    - Load YOUR financial dataset instead of rotten_tomatoes
    - Use model="ProsusAI/finbert" as base (better for finance)
    - num_labels=3 (positive/negative/neutral)
    - Evaluate with F1 per class (not just overall)

  ToxicContentDetection:
    Same workflow, but:
    - Use a toxicity dataset (or your own labeled data)
    - num_labels depends on your categories
    - Consider multi-label classification if a text can be
      toxic in multiple ways (threat + insult)

  COMPLETE LEARNING PATH DONE:
    Step 1: Tokenization        ✓ (text → numbers)
    Step 2: Embeddings          ✓ (numbers → vectors)
    Step 3: Self-Attention      ✓ (context understanding)
    Step 4: Transformer Block   ✓ (full architecture)
    Step 5: Inference Pipeline  ✓ (use pretrained models)
    Step 6: Fine-Tuning         ✓ (train on your data) ← DONE

  You now have the complete foundation to build any NLP project.
""")

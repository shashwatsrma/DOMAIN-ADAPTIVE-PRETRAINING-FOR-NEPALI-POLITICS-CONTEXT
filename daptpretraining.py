# ============================================================
# DAPT_PRETRAINING.PY — Domain Adaptive Pretraining for BERT
# Adapts bert-base-uncased to Nepali English news domain
# Updated for ~49,399 articles from TKP, Setopati, Ratopati,
# OnlineKhabar, MyRepublica + factcheck corpus
# ============================================================

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ==================== DEVICE AUTO-DETECT ====================
if torch.cuda.is_available():
    device = "cuda"
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Silicon (MPS)")
else:
    device = "cpu"
    print("⚠️  No GPU found — using CPU")

# ==================== CONFIG ====================
# CHANGED: batch sizes increased for larger corpus
# CHANGED: grad_accum reduced on cuda since batch is bigger
DEVICE_CONFIG = {
    "cuda": {"batch_size": 64, "grad_accum": 2},   # was 32/2
    "mps":  {"batch_size": 16, "grad_accum": 4},   # unchanged — MPS memory limit
    "cpu":  {"batch_size": 8,  "grad_accum": 8},   # unchanged
}

BATCH_SIZE = DEVICE_CONFIG[device]["batch_size"]
GRAD_ACCUM = DEVICE_CONFIG[device]["grad_accum"]

MAX_LENGTH     = 256
EPOCHS         = 4          # CHANGED: was 2 — 4 is the sweet spot for 49k articles
MLM_PROB       = 0.15
SEED           = 42
OUTPUT_DIR     = "./bert_nepali_dapt"
DATA_PATH      = "Nepali_Politics.csv"
FACTCHECK_PATH = "factcheck.csv"

# CHANGED: added eval split so we can track MLM loss per epoch
# and stop early if loss plateaus
EVAL_SPLIT     = 0.05       # 5% of corpus used for validation

print(f"\nConfig → Batch: {BATCH_SIZE} | Grad Accum: {GRAD_ACCUM} | "
      f"Effective Batch: {BATCH_SIZE * GRAD_ACCUM} | Epochs: {EPOCHS}")

# ==================== ENCODING FIX ====================
def fix_encoding(s):
    try:
        return str(s).encode("latin1").decode("utf-8", errors="ignore")
    except Exception:
        return str(s)

# ==================== HELPER : FIND TITLE/BODY ====================
def find_title_body_columns(df, name="dataset"):
    title_col = None
    body_col  = None

    for c in df.columns:
        lc = c.strip().lower()
        if lc == "title":
            title_col = c
        elif lc in ("body", "content", "text"):
            body_col = c

    if title_col is None or body_col is None:
        raise ValueError(
            f"{name}: TITLE/BODY columns not found.\n"
            f"Columns are: {df.columns.tolist()}"
        )

    print(f"  {name}: title='{title_col}'  body='{body_col}'")
    return title_col, body_col

# ==================== LOAD DATA ====================
print("\n" + "=" * 55)
print("LOADING DATA")
print("=" * 55)

# ── Main true-news file ──────────────────────────────────────
print("\n[1/2] Loading main dataset...")
df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
title_col, body_col = find_title_body_columns(df, "Main CSV")

df["title_text"] = df[title_col].fillna("").apply(fix_encoding).str.strip()
df["body_text"]  = df[body_col].fillna("").apply(fix_encoding).str.strip()

print(f"  Articles      : {len(df):,}")
print(f"  Missing body  : {(df['body_text']  == '').sum():,}")
print(f"  Missing title : {(df['title_text'] == '').sum():,}")

# ── Fact-check file ──────────────────────────────────────────
print("\n[2/2] Loading fact-check dataset...")
fc_df = pd.read_csv(FACTCHECK_PATH, encoding="latin1", low_memory=False)
fc_title_col, fc_body_col = find_title_body_columns(fc_df, "Fact-check CSV")

fc_df["title_text"] = fc_df[fc_title_col].fillna("").apply(fix_encoding).str.strip()
fc_df["body_text"]  = fc_df[fc_body_col].fillna("").apply(fix_encoding).str.strip()

print(f"  Fact-check articles: {len(fc_df):,}")

# ==================== BUILD TEXT RECORDS ====================
# CHANGED: body truncation raised from 1500 → 2000 chars
# because the larger corpus can afford richer context windows
# and 49k articles means BERT sees enough variety even with
# longer sequences. Keep MAX_LENGTH=256 tokens as the hard cap.

def extract_records(frame):
    out = []
    for _, row in frame.iterrows():
        title = row["title_text"]
        body  = row["body_text"]

        # Always add title as a standalone sequence
        # BERT learns short, punchy Nepali news vocabulary from these
        if len(title) > 10:
            out.append(title)

        # Add title + body combined (richer context)
        if len(body) > 10:
            combined = f"{title} {body}".strip()
            out.append(combined[:2000])          # was 1500

    return out

true_records = extract_records(df)
fc_records   = extract_records(fc_df)

all_records  = true_records + fc_records
all_records  = list(set(r for r in all_records if len(r) > 10))

print(f"\n{'=' * 55}")
print("CORPUS SUMMARY")
print(f"{'=' * 55}")
print(f"  True news sequences   : ~{len(true_records):,}")
print(f"  Fact-check sequences  : ~{len(fc_records):,}")
print(f"  After deduplication   :  {len(all_records):,}")

# ==================== TRAIN / EVAL SPLIT ====================
# CHANGED: added eval split — didn't exist before
# Lets us track MLM loss on held-out data each epoch
# so we can see if pretraining is converging or diverging

import random
random.seed(SEED)
random.shuffle(all_records)

n_eval   = max(500, int(len(all_records) * EVAL_SPLIT))
n_train  = len(all_records) - n_eval

train_records = all_records[:n_train]
eval_records  = all_records[n_train:]

print(f"\n  Train sequences : {len(train_records):,}")
print(f"  Eval  sequences : {len(eval_records):,}")

# ==================== TOKENIZATION ====================
print("\n" + "=" * 55)
print("TOKENIZING")
print("=" * 55)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = Dataset.from_dict({"text": train_records})
eval_dataset  = Dataset.from_dict({"text": eval_records})

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True,
    )

tokenized_train = train_dataset.map(
    tokenize_fn, batched=True, batch_size=1000,
    remove_columns=["text"], desc="Tokenizing train"
)
tokenized_eval = eval_dataset.map(
    tokenize_fn, batched=True, batch_size=1000,
    remove_columns=["text"], desc="Tokenizing eval"
)

print(f"✅ Train: {len(tokenized_train):,} sequences")
print(f"✅ Eval : {len(tokenized_eval):,} sequences")

# ==================== MODEL ====================
print("\nLoading BERT for MLM...")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# ==================== DATA COLLATOR ====================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROB,
)

# ==================== TRAINING ARGUMENTS ====================
use_fp16 = (device == "cuda")
use_bf16 = (device == "mps")

# CHANGED: logging_steps reduced from 100 → 50
#          so you see progress more often on the larger corpus
# CHANGED: eval_strategy added — evaluates every epoch
# CHANGED: load_best_model_at_end=True — saves the epoch
#          with lowest eval MLM loss, not just the last epoch
# CHANGED: save_steps increased from 1000 → 2000
#          because the larger corpus has more steps per epoch
# CHANGED: warmup_ratio kept at 0.06 — appropriate for 4 epochs

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,

    fp16=use_fp16,
    bf16=use_bf16,

    logging_steps=50,
    save_strategy="epoch",          # FIXED: was save_steps=2000
    save_total_limit=2,

    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,

    seed=SEED,
    dataloader_num_workers=2,
    report_to="none",
)
# ==================== TRAINER ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,    # CHANGED: added eval dataset
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ==================== TRAIN ====================
print("\n" + "=" * 55)
print("STARTING DAPT")
print("=" * 55)
print(f"  Corpus       : {len(all_records):,} sequences")
print(f"  Epochs       : {EPOCHS}")
print(f"  Device       : {device.upper()}")
print(f"  Eff. batch   : {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Learning rate: 3e-5")
print(f"  Eval         : every epoch (saves best MLM loss)\n")

trainer.train()

# ==================== SANITY CHECK ====================
# CHANGED: added fill-mask test after training
# Lets you verify BERT learned Nepali political domain
# before moving to the fake news fine-tuning step

print("\n" + "=" * 55)
print("POST-DAPT SANITY CHECK (fill-mask)")
print("=" * 55)

try:
    from transformers import pipeline
    fill = pipeline("fill-mask", model=model, tokenizer=tokenizer,
                    device=0 if device == "cuda" else -1)

    test_sentences = [
        "Prime Minister [MASK] dissolved the parliament.",
        "The no-confidence [MASK] was filed against the government.",
        "CPN-UML and [MASK] Centre agreed to form a coalition.",
        "The Election Commission announced [MASK] elections.",
        "Nepali Congress leader [MASK] won the confidence vote.",
    ]

    for sent in test_sentences:
        results = fill(sent)
        top3 = [r["token_str"] for r in results[:3]]
        print(f"  '{sent}'")
        print(f"   → Top 3: {top3}\n")

except Exception as e:
    print(f"  Sanity check skipped: {e}")

# ==================== SAVE ====================
print(f"\nSaving model to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ DAPT COMPLETE")
print(f"   Saved to : {OUTPUT_DIR}")
print(f"\nUse in your classifier:")
print(f'   BertTokenizer.from_pretrained("{OUTPUT_DIR}")')
print(f'   BertModel.from_pretrained("{OUTPUT_DIR}")')
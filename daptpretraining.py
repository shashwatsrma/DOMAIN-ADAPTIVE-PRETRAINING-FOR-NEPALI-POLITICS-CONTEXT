# ============================================================
# DAPT_PRETRAINING.PY — Domain Adaptive Pretraining for BERT
# Adapts bert-base-uncased to Nepali English news domain
# using Masked Language Modeling (MLM) on TITLE + BODY text
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
DEVICE_CONFIG = {
    "cuda": {"batch_size": 32, "grad_accum": 2},
    "mps":  {"batch_size": 16, "grad_accum": 4},
    "cpu":  {"batch_size": 8,  "grad_accum": 8},
}

BATCH_SIZE = DEVICE_CONFIG[device]["batch_size"]
GRAD_ACCUM = DEVICE_CONFIG[device]["grad_accum"]

MAX_LENGTH     = 256
EPOCHS         = 2
MLM_PROB       = 0.15
SEED           = 42
OUTPUT_DIR     = "./bert_nepali_dapt"
DATA_PATH      = "finaltrue.csv"
FACTCHECK_PATH = "factcheck.csv"

print(f"\nConfig → Batch: {BATCH_SIZE} | Grad Accum: {GRAD_ACCUM} | Effective Batch: {BATCH_SIZE * GRAD_ACCUM}")

# ==================== ENCODING FIX ====================
def fix_encoding(s):
    try:
        return str(s).encode("latin1").decode("utf-8", errors="ignore")
    except Exception:
        return str(s)

# ==================== HELPER : FIND TITLE/BODY ====================
def find_title_body_columns(df, name="dataset"):
    title_col = None
    body_col = None

    for c in df.columns:
        lc = c.strip().lower()
        if lc == "title":
            title_col = c
        elif lc == "body" or lc == "content" or lc == "text":
            body_col = c

    if title_col is None or body_col is None:
        raise ValueError(
            f"{name}: TITLE/BODY columns not found.\nColumns are: {df.columns.tolist()}"
        )

    print(f"{name}: using title column = {title_col}")
    print(f"{name}: using body  column = {body_col}")

    return title_col, body_col

# ==================== LOAD DATA ====================
print("\n" + "="*50)
print("LOADING DATA")
print("="*50)

# -------- main true-news file (NO LABEL needed) --------
print("\n[1/2] Loading main dataset...")
df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)

title_col, body_col = find_title_body_columns(df, "Main CSV")

true_df = df.copy()

true_df["title_text"] = true_df[title_col].fillna("").apply(fix_encoding).str.strip()
true_df["body_text"]  = true_df[body_col].fillna("").apply(fix_encoding).str.strip()

missing_body  = (true_df["body_text"] == "").sum()
missing_title = (true_df["title_text"] == "").sum()

print(f"  Articles:         {len(true_df):,}")
print(f"  Missing BODY:     {missing_body:,}")
print(f"  Missing TITLE:    {missing_title:,}")

# -------- fact-check file --------
print("\n[2/2] Loading fact-check dataset...")
fc_df = pd.read_csv(FACTCHECK_PATH, encoding="latin1", low_memory=False)

fc_title_col, fc_body_col = find_title_body_columns(fc_df, "Fact-check CSV")

fc_df["title_text"] = fc_df[fc_title_col].fillna("").apply(fix_encoding).str.strip()
fc_df["body_text"]  = fc_df[fc_body_col].fillna("").apply(fix_encoding).str.strip()

print(f"  Fact-check articles: {len(fc_df):,}")

# ==================== BUILD RECORDS ====================
def extract_records(frame):
    out = []
    for _, row in frame.iterrows():
        title = row["title_text"]
        body  = row["body_text"]

        if len(title) > 10:
            out.append(title)

        if len(body) > 10:
            combined = f"{title} {body}".strip()
            out.append(combined[:1500])

    return out

true_records = extract_records(true_df)
fc_records   = extract_records(fc_df)

all_records = true_records + fc_records
all_records = list(set([r for r in all_records if len(r) > 10]))

print(f"\n{'='*50}")
print("CORPUS SUMMARY")
print(f"{'='*50}")
print(f"  True news sequences:   ~{len(true_records):,}")
print(f"  Fact-check sequences:  ~{len(fc_records):,}")
print(f"  After deduplication:    {len(all_records):,}")

# ==================== TOKENIZATION ====================
print("\n" + "="*50)
print("TOKENIZING")
print("="*50)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = Dataset.from_dict({"text": all_records})

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True,
    )

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    desc="Tokenizing",
)

print(f"✅ Tokenized {len(tokenized):,} sequences")

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

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,

    fp16=use_fp16,
    bf16=use_bf16,

    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,

    seed=SEED,
    dataloader_num_workers=2,
    report_to="none",
)

# ==================== TRAINER ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    processing_class=tokenizer,   # ← new name
)

# ==================== TRAIN ====================
print("\n" + "="*50)
print("STARTING DAPT")
print("="*50)

print(f"Corpus:          {len(all_records):,}")
print(f"Epochs:          {EPOCHS}")
print(f"Device:          {device.upper()}")
print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM}\n")

trainer.train()

# ==================== SAVE ====================
print(f"\nSaving model to {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ DAPT COMPLETE")
print(f"Saved to: {OUTPUT_DIR}")

print("\nUse in your classifier:")
print(f'BertTokenizer.from_pretrained("{OUTPUT_DIR}")')
print(f'BertModel.from_pretrained("{OUTPUT_DIR}")')
"""
get_data.py

Loads IITB Hindi-English dataset, samples train/val/test splits,
flattens translation entries, and saves both native and romanized JSONL
files ready for fine-tuning.
"""

import os
import json
from datasets import load_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Config
NUM_TRAIN = 10_000
NUM_VAL   = 1_000
SEED_TRAIN = 42
SEED_VAL   = 123

NATIVE_DIR   = "datasets/native"
ROMAN_DIR    = "datasets/romanized"

# Utility Functions
def flatten_translation(example):
    """
    Handles both dict or list in 'translation' field.
    Returns top-level {'hi', 'en'}.
    """
    translations = example["translation"]
    if isinstance(translations, list):
        translations = translations[0]  # take first entry if list
    return {
        "hi": translations["hi"],
        "en": translations["en"]
    }

def save_jsonl(dataset, path):
    """Save a HuggingFace dataset to JSONL with hi/en fields."""
    with open(path, "w", encoding="utf8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {path} ({len(dataset)} rows)")

def transliterate_file(infile, outfile):
    """Convert Hindi text in JSONL to Romanized ITRANS."""
    with open(infile, "r", encoding="utf8") as fin, \
         open(outfile, "w", encoding="utf8") as fout:
        for line in fin:
            obj = json.loads(line)
            obj["hi"] = transliterate(obj["hi"], sanscript.DEVANAGARI, sanscript.ITRANS).lower()
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Romanized file saved: {outfile}")

# 1. Load IITB dataset
print("Loading IITB Hindi-English dataset...")
ds = load_dataset("cfilt/iitb-english-hindi")

print("Dataset splits:", ds.keys())
for split_name, split in ds.items():
    print(f"{split_name} rows:", len(split))

# 2. Sample train/val/test
print(f"Sampling {NUM_TRAIN} train rows...")
train_split = ds["train"].shuffle(seed=SEED_TRAIN).select(range(NUM_TRAIN))
print(f"Sampling {NUM_VAL} validation rows...")
val_split   = ds["train"].shuffle(seed=SEED_VAL).select(range(NUM_VAL))
test_split  = ds["test"]  # full test set

# 3. Flatten translation column
print("Flattening translation fields...")
train_split = train_split.map(flatten_translation, remove_columns=["translation"])
val_split   = val_split.map(flatten_translation, remove_columns=["translation"])
test_split  = test_split.map(flatten_translation, remove_columns=["translation"])

# 4. Save native JSONL
os.makedirs(NATIVE_DIR, exist_ok=True)
save_jsonl(train_split, os.path.join(NATIVE_DIR, "train.jsonl"))
save_jsonl(val_split,   os.path.join(NATIVE_DIR, "val.jsonl"))
save_jsonl(test_split,  os.path.join(NATIVE_DIR, "test.jsonl"))

# 5. Save romanized JSONL
os.makedirs(ROMAN_DIR, exist_ok=True)
transliterate_file(os.path.join(NATIVE_DIR, "train.jsonl"),
                   os.path.join(ROMAN_DIR, "train.jsonl"))
transliterate_file(os.path.join(NATIVE_DIR, "val.jsonl"),
                   os.path.join(ROMAN_DIR, "val.jsonl"))
transliterate_file(os.path.join(NATIVE_DIR, "test.jsonl"),
                   os.path.join(ROMAN_DIR, "test.jsonl"))
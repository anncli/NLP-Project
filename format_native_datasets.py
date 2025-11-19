"""
format_native_datasets.py

This script processes the IITB Hindi→English parallel corpus from HuggingFace
(cfilt/iitb-english-hindi) and saves it in a flattened JSONL format with
columns "hi" and "en" (Hindi first, English second).
- Samples a configurable number of training rows (default 100,000)
- Keeps full validation and test sets
- Removes the nested "translation" column
- Saves each split in a "datasets/" folder

Usage:
- python3 format_native_datasets.py
- If `from datasets import load_dataset` fails, try `pip install requirements.txt`
"""

from datasets import load_dataset
import json
import os

# -------------------------------------------------------
# 1. Load IITB English–Hindi dataset from HuggingFace
# -------------------------------------------------------
print("Loading IITB dataset...")
ds = load_dataset("cfilt/iitb-english-hindi")

# -------------------------------------------------------
# 2. Sample rows from each split
# -------------------------------------------------------
print("Sampling rows from each split...")
NUM_TRAIN_ROWS = 100_000
small_train = ds["train"].select(range(NUM_TRAIN_ROWS))
small_val   = ds["validation"]  # full validation set
small_test  = ds["test"]        # full test set

# -------------------------------------------------------
# 3. Flatten translation column and keep 'hi' first
# -------------------------------------------------------
def flatten_translation(example):
    return {
        "hi": example["translation"]["hi"],
        "en": example["translation"]["en"],
    }

gsm_train = small_train.map(flatten_translation, remove_columns=["translation"])
gsm_val   = small_val.map(flatten_translation, remove_columns=["translation"])
gsm_test  = small_test.map(flatten_translation, remove_columns=["translation"])

# -------------------------------------------------------
# 4. Save each split as JSONL in datasets/ folder
# -------------------------------------------------------
output_folder = "datasets"
os.makedirs(output_folder, exist_ok=True)

def save_jsonl(dataset, path):
    print(f"Saving {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            # Write hi first, then en
            f.write(json.dumps({"hi": item["hi"], "en": item["en"]}, ensure_ascii=False) + "\n")

save_jsonl(gsm_train, f"{output_folder}/native_train.jsonl")
save_jsonl(gsm_val, f"{output_folder}/native_val.jsonl")
save_jsonl(gsm_test, f"{output_folder}/native_test.jsonl")

print("Done! Files saved in 'datasets' folder.")
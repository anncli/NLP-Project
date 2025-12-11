import os
import json
import random

# Settings
SEED = 42
random.seed(SEED)

native_folder = "../datasets/native"
romanized_folder = "../datasets/romanized_new"
combined_folder = "../datasets/combined"

os.makedirs(combined_folder, exist_ok=True)

splits = ["train", "test", "val"]

# -----------------------------
# Combine files per split
# -----------------------------
for split in splits:
    all_examples = []

    for folder in [native_folder, romanized_folder]:
        file_path = os.path.join(folder, f"{split}.jsonl")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_examples.extend([json.loads(line) for line in f])

    # Shuffle examples with seed
    random.shuffle(all_examples)

    # Write to combined folder
    combined_path = os.path.join(combined_folder, f"{split}.jsonl")
    with open(combined_path, "w") as f:
        for ex in all_examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

    print(f"Combined {len(all_examples)} examples for {split} -> {combined_path}")

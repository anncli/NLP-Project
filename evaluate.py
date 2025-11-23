"""
BLEU Evaluation Script
----------------------
This script computes BLEU scores for two translation systems:
1) Model using Devanagari Hindi input
2) Model using Romanized Hindi input

Required file format (plain .txt files):
- data/references.txt      → gold English translation, one sentence per line
- outputs/preds_deva.txt   → model outputs from Devanagari input, one sentence per line
- outputs/preds_roman.txt  → model outputs from Romanized input, one sentence per line
"""

import sacrebleu

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# Load data
references = load_lines("data/references.txt")
preds_deva = load_lines("outputs/preds_deva.txt")
preds_roman = load_lines("outputs/preds_roman.txt")

# Basic sanity check
assert len(references) == len(preds_deva) == len(preds_roman), "Line count mismatch between files."

# Compute BLEU for Devanagari-input model
bleu_deva = sacrebleu.corpus_bleu(preds_deva, [references])
print("BLEU score (Devanagari input):", bleu_deva.score)

# Compute BLEU for Romanized-input model
bleu_roman = sacrebleu.corpus_bleu(preds_roman, [references])
print("BLEU score (Romanized input):", bleu_roman.score)

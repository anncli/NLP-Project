import os
import json
import torch
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])

from ai4bharat.transliteration import XlitEngine

NATIVE_DIR = "datasets/native"
ROMAN_NEW_DIR = "datasets/romanized_new"

engine = XlitEngine(src_script_type="indic", beam_width=4, rescore=False)

def transliterate_ai4bharat(infile, outfile):
    translit_word = engine.translit_word

    with open(infile, "r", encoding="utf8") as fin, \
         open(outfile, "w", encoding="utf8") as fout:

        for line in fin:
            obj = json.loads(line)
            text = obj["hi"]

            words = text.split(" ")
            out_words = []

            for w in words:
                try:
                    result = translit_word(w, lang_code="hi", topk=1)
                    out_words.append(result[0])
                except Exception:
                    out_words.append(w)

            obj["hi"] = " ".join(out_words)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"AI4Bharat romanized file saved: {outfile}")


os.makedirs(ROMAN_NEW_DIR, exist_ok=True)

files = ["train.jsonl", "val.jsonl", "test.jsonl"]

for fname in files:
    src = os.path.join(NATIVE_DIR, fname)
    tgt = os.path.join(ROMAN_NEW_DIR, fname)

    if not os.path.exists(src):
        print(f"Skipping {fname}: native file missing")
        continue

    print(f"Romanizing {fname}...")
    transliterate_ai4bharat(src, tgt)

print("Done!")

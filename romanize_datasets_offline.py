import json
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

input_path = "datasets/test_native.jsonl"
output_path = "datasets/test_romanized_ITRANS.jsonl"

with open(input_path, "r", encoding="utf8") as infile, \
     open(output_path, "w", encoding="utf8") as outfile:

    for line in infile:
        obj = json.loads(line)
        # Devanagari → HK (ASCII-friendly)
        romanized_hi = transliterate(obj["hi"], sanscript.DEVANAGARI, sanscript.ITRANS)
        # Lowercase for casual typing
        romanized_hi = romanized_hi.lower()
        # Simplify doubled vowels to mimic casual typing
        romanized_hi = romanized_hi.replace("aa", "a").replace("ii", "i").replace("uu", "u")
        obj["hi"] = romanized_hi
        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Done! Romanized file saved to:", output_path)

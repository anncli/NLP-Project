import json
import requests
import urllib.parse

API_URL = "http://aksharamukha-plugin.appspot.com/api/public"

def romanize_hi(text):
    params = {
        "source": "Devanagari",
        "target": "RomanColloquial",
        "text": text
    }
    encoded = urllib.parse.urlencode(params)
    url = f"{API_URL}?{encoded}"

    response = requests.get(url)
    
    # Debug: if response is empty or HTML
    if not response.text.strip():
        raise ValueError("Empty response from API")

    # Some Aksharamukha responses are plain text, not JSON
    try:
        data = response.json()
        return data["text"]
    except json.JSONDecodeError:
        # If not JSON, return raw text
        return response.text.strip()

input_path = "datasets/test_native.jsonl"
output_path = "datasets/test_romanized_RomanColloquial.jsonl"

with open(input_path, "r", encoding="utf8") as infile, \
     open(output_path, "w", encoding="utf8") as outfile:

    for line in infile:
        obj = json.loads(line)
        obj["hi"] = romanize_hi(obj["hi"]).lower()
        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Done! Romanized file saved to:", output_path)

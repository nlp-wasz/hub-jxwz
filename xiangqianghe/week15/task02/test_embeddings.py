import requests
import sys

models = ["qwen:0.5b", "qwen3:0.6b"]

for model in models:
    print(f"Testing model: {model}")
    try:
        resp = requests.post('http://localhost:11434/api/embeddings', json={"model": model, "prompt": "test"})
        if resp.status_code == 200:
            print(f"SUCCESS: {model} supports embeddings.")
        else:
            print(f"FAILURE: {model} returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"ERROR: {e}")

try:
    import sentence_transformers
    print("sentence_transformers is installed.")
except ImportError:
    print("sentence_transformers is NOT installed.")

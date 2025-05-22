import requests, json

def ollama_generate(model: str,
                    prompt: str,
                    url: str = "http://localhost:11434/api/generate") -> str:
    resp = requests.post(url, json={"model": model, "prompt": prompt}, stream=False)
    text = resp.text.strip().splitlines()
    pieces = []
    for line in text:
        try:
            obj = json.loads(line)
            pieces.append(obj.get("response", ""))
        except json.JSONDecodeError:
            continue
    return "".join(pieces).strip()

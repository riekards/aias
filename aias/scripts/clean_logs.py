import json

input_path  = "memory/logs.jsonl"
output_path = "memory/logs_clean.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # ensure it has both user and ai keys
            if "user" in obj and "ai" in obj:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            # skip any malformed lines
            continue

print(f"Cleaned logs written to {output_path}")

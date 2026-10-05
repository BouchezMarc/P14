import json
from collections import Counter

for filename in [
    "data/processed/anonymized/dpo_train.jsonl",
    "data/processed/anonymized/dpo_validation.jsonl",
]:
    counts = Counter()

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts[row["source"]["label_type"]] += 1

    total = sum(counts.values())

    print(f"\n{filename}")
    print("=" * 50)
    print(f"Total : {total}")

    for label, count in sorted(counts.items()):
        print(f"{label:10s}: {count:5d} ({count / total:.2%})")
import json
from collections import defaultdict

path = "data/processed/medical_rewardbench.jsonl"

groups = defaultdict(list)

with open(path, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        row = json.loads(line)

        key = (
            row.get("prompt"),
            row.get("chosen"),
            row.get("rejected"),
        )

        groups[key].append((line_no, row))

duplicates = [
    items
    for items in groups.values()
    if len(items) > 1
]

print("=" * 80)
print(f"Doublons exacts trouvés : {len(duplicates)}")
print("=" * 80)

for i, items in enumerate(duplicates, 1):
    print(f"\nDOUBLON #{i}")
    print("-" * 80)

    for line_no, row in items:
        print(f"\n--- Ligne {line_no} ---")
        print(json.dumps(row, ensure_ascii=False, indent=2))

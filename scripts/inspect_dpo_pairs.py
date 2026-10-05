import json
from collections import defaultdict

FILE = "data/processed/anonymized/dpo_train.jsonl"
groups = defaultdict(list)

with open(FILE, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = r["source"]["label_type"]
        if len(groups[t]) < 3:
            c = r["chosen"][-1]["content"]
            x = r["rejected"][-1]["content"]
            groups[t].append((len(c), len(x), c[:300], x[:300]))

for t in ["easy", "hard", "length"]:
    print(f"\n{'='*60}\n{t.upper()}")
    for i, (lc, lr, c, r) in enumerate(groups[t], 1):
        print(f"\n{i}. chosen={lc} chars | rejected={lr} chars")
        print(f" C: {c}")
        print(f" R: {r}")
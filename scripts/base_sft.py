import json

TEST_FILE = "data/processed/sft_test.jsonl"
BASE_FILE = "results/base/predictions.jsonl"
SFT_FILE = "results/sft_v5/predictions.jsonl"

N = 20


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


test = load_jsonl(TEST_FILE)
base = load_jsonl(BASE_FILE)
sft = load_jsonl(SFT_FILE)

base_by_id = {x["id"]: x for x in base}
sft_by_id = {x["id"]: x for x in sft}

cases = []

for item in test:
    id_ = item["id"]

    if id_ not in base_by_id or id_ not in sft_by_id:
        continue

    b = base_by_id[id_]
    s = sft_by_id[id_]

    base_pred = b["prediction"]
    sft_pred = s["prediction"]

    if base_pred.strip() != sft_pred.strip():
        cases.append({
            "id": id_,
            "instruction": item["instruction"],
            "reference": item["response"],
            "base": base_pred,
            "sft": sft_pred,
        })


print("=" * 80)
print(f"EXEMPLES DIFFÉRENTS BASE / SFT v5 : {len(cases)}")
print("=" * 80)

for i, case in enumerate(cases[:N], 1):
    print(f"\n{'=' * 80}")
    print(f"EXEMPLE {i}/{min(N, len(cases))}")
    print(f"ID : {case['id']}")
    print(f"{'=' * 80}")

    print("\nQUESTION :")
    print(case["instruction"])

    print("\nRÉFÉRENCE :")
    print(case["reference"])

    print("\nBASE :")
    print(case["base"])

    print("\nSFT v4 :")
    print(case["sft"])
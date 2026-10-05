import json

TEST_FILE = "data/processed/sft_test.jsonl"
OUTPUT_FILE = "results/sft_v4/critical_test_examples.jsonl"

IDS = {
    "7861dadabd5919ec",
    "eabc18f9c3d45f07",
    "9df4002842ead743",
    "7440afd91f29c90d",
    "4e910ff424b36c7d",
    "4f79baf6ad5403b8",
    "9f0ba089d79d4590",
    "e2d9fe16f34723b0",
    "c3c15891df519229",
    "c9ed03c11d131b3f",
    "41f3ca86b75ef4b7",
    "7e1ef1d8e1432864",
    "311d5ae9de58be2a",
    "e9326c4bbd6436b4",
    "b4e0e0c97b75d084",
    "25f87711ecc924af",
    "141f3469b508be4e",
    "180f6faa485d045a",
    "41b986eeefc4aea8",
    "37e0fd07263a5680",
}

found = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        if record.get("id") in IDS:
            found.append(record)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in found:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Demandés : {len(IDS)}")
print(f"Trouvés  : {len(found)}")
print(f"Sortie   : {OUTPUT_FILE}")

missing = IDS - {r["id"] for r in found}

if missing:
    print("\nIDs absents du test :")
    for id_ in sorted(missing):
        print(id_)
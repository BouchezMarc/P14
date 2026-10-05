import json

TRAIN_FILE = "data/processed/sft_train.jsonl"

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    shown = 0

    for line in f:
        record = json.loads(line)

        if record.get("task") != "medical_mcq":
            continue

        source_specific = record.get("source_specific", {})

        if source_specific.get("question_type") != "multiple":
            continue

        print("\n" + "=" * 100)
        print("SOURCE :", record.get("source", {}).get("dataset"))
        print("ID     :", record.get("id"))
        print("CORRECT:", source_specific.get("correct_answers"))

        print("\nQUESTION:")
        print(record.get("instruction"))

        print("\nRESPONSE:")
        print(record.get("response"))

        shown += 1

        if shown >= 10:
            break

print(f"\n{shown} exemples affichés.")
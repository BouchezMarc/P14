import json
import re
from pathlib import Path
from collections import defaultdict

PATH = Path(r"data/processed/sft_5000.jsonl")


# ============================================================
# CHARGEMENT
# ============================================================

rows = []

with PATH.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if line.strip():
            rows.append((line_no, json.loads(line)))


# ============================================================
# 1. DOUBLONS
# ============================================================

print("=" * 80)
print("DOUBLONS DE QUESTIONS")
print("=" * 80)

duplicates = defaultdict(list)

for line_no, row in rows:

    instruction = str(
        row.get("instruction", "")
    ).strip().lower()

    duplicates[instruction].append(
        (line_no, row)
    )


duplicate_groups = {
    question: entries
    for question, entries in duplicates.items()
    if len(entries) > 1
}

print(
    f"\nGroupes de doublons : {len(duplicate_groups)}"
)

for i, (question, entries) in enumerate(
    duplicate_groups.items(),
    1
):

    print("\n" + "-" * 80)
    print(f"DUPLICAT #{i}")
    print(f"QUESTION : {question}")

    for line_no, row in entries:

        print(
            f"\n  Ligne      : {line_no}"
            f"\n  ID         : {row.get('id')}"
            f"\n  Source     : {row.get('source', {}).get('dataset')}"
            f"\n  Original ID: {row.get('source', {}).get('original_id')}"
            f"\n  Task       : {row.get('task')}"
            f"\n  Langue     : {row.get('language')}"
        )

        print(
            f"  Réponse    : "
            f"{str(row.get('response', ''))[:500]}"
        )


# ============================================================
# 2. EMAILS
# ============================================================

print("\n\n" + "=" * 80)
print("EMAILS DETECTÉS")
print("=" * 80)

email_pattern = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

email_found = []

for line_no, row in rows:

    text_fields = {
        "instruction": str(
            row.get("instruction", "")
        ),
        "response": str(
            row.get("response", "")
        ),
    }

    for field, text in text_fields.items():

        matches = email_pattern.findall(text)

        for email in matches:

            email_found.append(
                (
                    line_no,
                    row,
                    field,
                    email
                )
            )


print(
    f"\nEmails détectés : {len(email_found)}"
)

for line_no, row, field, email in email_found:

    print("\n" + "-" * 80)

    print(
        f"Ligne       : {line_no}"
        f"\nID          : {row.get('id')}"
        f"\nSource      : {row.get('source', {}).get('dataset')}"
        f"\nOriginal ID : {row.get('source', {}).get('original_id')}"
        f"\nChamp       : {field}"
        f"\nEmail       : {email}"
    )

    text = str(row.get(field, ""))

    position = text.find(email)

    start = max(0, position - 150)
    end = min(len(text), position + len(email) + 150)

    print(
        f"\nContexte :\n{text[start:end]}"
    )


# ============================================================
# 3. INSTRUCTIONS TRES COURTES
# ============================================================

print("\n\n" + "=" * 80)
print("INSTRUCTIONS TRES COURTES")
print("=" * 80)

short_found = []

for line_no, row in rows:

    instruction = str(
        row.get("instruction", "")
    ).strip()

    if len(instruction) < 10:

        short_found.append(
            (line_no, row)
        )


print(
    f"\nInstructions < 10 caractères : "
    f"{len(short_found)}"
)

for line_no, row in short_found:

    print("\n" + "-" * 80)

    print(
        f"Ligne       : {line_no}"
        f"\nID          : {row.get('id')}"
        f"\nSource      : {row.get('source', {}).get('dataset')}"
        f"\nTask        : {row.get('task')}"
        f"\nInstruction : {repr(row.get('instruction'))}"
        f"\nResponse    : {str(row.get('response', ''))[:300]}"
    )
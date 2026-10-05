import json
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("data/processed/sft_5000_anonymized.jsonl")


def normalize(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return " ".join(value.strip().lower().split())

    return str(value).strip().lower()


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Ligne {line_number} invalide : {exc}"
                ) from exc

    return records


def get_value(record, *paths):
    """
    Essaie plusieurs chemins possibles dans le JSON.
    Exemple :
        get_value(record, ("language",), ("metadata", "language"))
    """

    for path in paths:
        current = record

        try:
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break

                current = current.get(key)

            if current not in (None, ""):
                return current

        except Exception:
            pass

    return None


def get_source(record):
    value = get_value(
        record,
        ("source",),
        ("metadata", "source"),
        ("provenance", "source"),
    )

    if isinstance(value, dict):
        value = (
            value.get("dataset")
            or value.get("name")
            or value.get("source")
        )

    return normalize(value)


def get_language(record):
    value = get_value(
        record,
        ("language",),
        ("lang",),
        ("metadata", "language"),
        ("source_specific", "language"),
        ("provenance", "language"),
    )

    return normalize(value)


def get_task(record):
    value = get_value(
        record,
        ("task",),
        ("type",),
        ("metadata", "task"),
        ("metadata", "type"),
        ("source_specific", "task"),
        ("source_specific", "type"),
    )

    return normalize(value)


def main():
    print("=" * 70)
    print("CONTROLE DE LA STRATIFICATION SFT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    records = load_jsonl(INPUT_FILE)

    print(f"\nFichier : {INPUT_FILE}")
    print(f"Nombre de records : {len(records)}")

    print("\n" + "=" * 70)
    print("1. EXEMPLE DE STRUCTURE")
    print("=" * 70)

    if records:
        first = records[0]

        print("\nClés racine :")
        for key in first.keys():
            print(f"  - {key}")

        print("\nValeurs utilisées pour la stratification :")
        print(f"  source   = {get_source(first)!r}")
        print(f"  language = {get_language(first)!r}")
        print(f"  task     = {get_task(first)!r}")

    print("\n" + "=" * 70)
    print("2. DISTRIBUTION SOURCE")
    print("=" * 70)

    source_counter = Counter()

    for record in records:
        source_counter[get_source(record)] += 1

    for value, count in source_counter.most_common():
        print(f"  {value!r:30} : {count}")

    print("\n" + "=" * 70)
    print("3. DISTRIBUTION LANGUAGE")
    print("=" * 70)

    language_counter = Counter()

    for record in records:
        language_counter[get_language(record)] += 1

    for value, count in language_counter.most_common():
        print(f"  {value!r:30} : {count}")

    print("\n" + "=" * 70)
    print("4. DISTRIBUTION TASK")
    print("=" * 70)

    task_counter = Counter()

    for record in records:
        task_counter[get_task(record)] += 1

    for value, count in task_counter.most_common():
        print(f"  {value!r:30} : {count}")

    print("\n" + "=" * 70)
    print("5. COMBINAISONS SOURCE × LANGUAGE × TASK")
    print("=" * 70)

    combinations = Counter()

    for record in records:
        source = get_source(record)
        language = get_language(record)
        task = get_task(record)

        combinations[(source, language, task)] += 1

    print(f"\nNombre de combinaisons : {len(combinations)}")

    for (source, language, task), count in sorted(combinations.items()):
        print(
            f"  source={source!r:20} "
            f"language={language!r:5} "
            f"task={task!r:20} "
            f"-> {count}"
        )

    print("\n" + "=" * 70)
    print("6. VALEURS MANQUANTES")
    print("=" * 70)

    missing_source = 0
    missing_language = 0
    missing_task = 0

    for record in records:
        if not get_source(record):
            missing_source += 1

        if not get_language(record):
            missing_language += 1

        if not get_task(record):
            missing_task += 1

    print(f"  Source manquante   : {missing_source}")
    print(f"  Language manquante : {missing_language}")
    print(f"  Task manquante     : {missing_task}")

    print("\n" + "=" * 70)
    print("7. CONTROLE FINAL")
    print("=" * 70)

    if len(combinations) == len(records):
        print(
            "⚠️ PROBLEME : chaque record possède une combinaison "
            "unique source × language × task."
        )
        print(
            "La stratification actuelle ne fonctionne donc pas "
            "comme prévu."
        )
    else:
        print(
            "✅ Les records partagent des groupes "
            "source × language × task."
        )

    if missing_source or missing_language or missing_task:
        print(
            "\n⚠️ Certaines valeurs de stratification sont absentes."
        )
    else:
        print(
            "\n✅ Source, language et task sont présents pour tous les records."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
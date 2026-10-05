from pathlib import Path
import json


DATASET_PATH = Path("data/processed/sft_5000.jsonl")

ALLOWED_LANGUAGES = {"fr", "en"}

ALLOWED_TASKS = {
    "medical_qa",
    "medical_mcq",
    "clinical_reasoning",
}

ALLOWED_DATASETS = {
    "MedQuAD",
    "FrenchMedMCQA",
    "MediQAl",
}

EXPECTED_TOP_LEVEL = {
    "id",
    "instruction",
    "response",
    "language",
    "task",
    "source",
    "clinical_context",
    "source_specific",
    "quality",
    "provenance",
}

EXPECTED_SOURCE = {
    "dataset",
    "original_id",
}

EXPECTED_CLINICAL_CONTEXT = {
    "clinical_case",
    "symptoms",
    "history",
    "vitals",
    "diagnosis",
    "medications",
    "allergies",
}

EXPECTED_SOURCE_SPECIFIC_COMMON = {
    "question_type",
    "question_focus",
    "synonyms",
    "umls_cui",
    "choices",
    "correct_answers",
}

EXPECTED_QUALITY = {
    "confidence",
    "validated",
}

EXPECTED_PROVENANCE = {
    "source_license",
    "processing_version",
}


def is_non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def validate_list_of_strings(value):
    return (
        isinstance(value, list)
        and all(isinstance(item, str) for item in value)
    )


def validate_record(record, line_number, seen_ids):
    errors = []

    # =========================================================
    # Niveau principal
    # =========================================================
    if not isinstance(record, dict):
        return [f"Ligne {line_number}: l'entrée n'est pas un objet JSON"]

    missing = EXPECTED_TOP_LEVEL - record.keys()

    if missing:
        errors.append(
            f"Ligne {line_number}: champs principaux manquants: "
            f"{sorted(missing)}"
        )

    # On ne considère pas les champs supplémentaires comme une erreur
    # afin de permettre l'évolution du schéma.

    # =========================================================
    # ID
    # =========================================================
    record_id = record.get("id")

    if not is_non_empty_string(record_id):
        errors.append(
            f"Ligne {line_number}: id absent ou vide"
        )
    else:
        if record_id in seen_ids:
            errors.append(
                f"Ligne {line_number}: ID technique dupliqué: {record_id}"
            )

        seen_ids.add(record_id)

    # =========================================================
    # Instruction
    # =========================================================
    if not is_non_empty_string(record.get("instruction")):
        errors.append(
            f"Ligne {line_number}: instruction absente ou vide"
        )

    # =========================================================
    # Response
    # =========================================================
    if not is_non_empty_string(record.get("response")):
        errors.append(
            f"Ligne {line_number}: response absente ou vide"
        )

    # =========================================================
    # Language
    # =========================================================
    language = record.get("language")

    if language not in ALLOWED_LANGUAGES:
        errors.append(
            f"Ligne {line_number}: language invalide: {language!r}"
        )

    # =========================================================
    # Task
    # =========================================================
    task = record.get("task")

    if task not in ALLOWED_TASKS:
        errors.append(
            f"Ligne {line_number}: task invalide: {task!r}"
        )

    # =========================================================
    # Source
    # =========================================================
    source = record.get("source")

    if not isinstance(source, dict):
        errors.append(
            f"Ligne {line_number}: source n'est pas un objet"
        )
    else:
        missing = EXPECTED_SOURCE - source.keys()

        if missing:
            errors.append(
                f"Ligne {line_number}: source champs manquants: "
                f"{sorted(missing)}"
            )

        dataset = source.get("dataset")

        if dataset not in ALLOWED_DATASETS:
            errors.append(
                f"Ligne {line_number}: dataset invalide: {dataset!r}"
            )

        if not is_non_empty_string(source.get("original_id")):
            errors.append(
                f"Ligne {line_number}: source.original_id absent ou vide"
            )

    # =========================================================
    # Clinical context
    # =========================================================
    clinical_context = record.get("clinical_context")

    if not isinstance(clinical_context, dict):
        errors.append(
            f"Ligne {line_number}: clinical_context n'est pas un objet"
        )
    else:
        missing = EXPECTED_CLINICAL_CONTEXT - clinical_context.keys()

        if missing:
            errors.append(
                f"Ligne {line_number}: clinical_context champs manquants: "
                f"{sorted(missing)}"
            )

        list_fields = [
            "symptoms",
            "history",
            "vitals",
            "medications",
            "allergies",
        ]

        for field in list_fields:
            if field in clinical_context:
                if not validate_list_of_strings(
                    clinical_context[field]
                ):
                    errors.append(
                        f"Ligne {line_number}: "
                        f"clinical_context.{field} doit être "
                        f"une liste de chaînes"
                    )

    # =========================================================
    # Source specific
    # =========================================================
    source_specific = record.get("source_specific")

    if not isinstance(source_specific, dict):
        errors.append(
            f"Ligne {line_number}: source_specific n'est pas un objet"
        )
    else:
        missing = (
            EXPECTED_SOURCE_SPECIFIC_COMMON
            - source_specific.keys()
        )

        if missing:
            errors.append(
                f"Ligne {line_number}: source_specific champs manquants: "
                f"{sorted(missing)}"
            )

        # -----------------------------------------------------
        # synonyms
        # -----------------------------------------------------
        if "synonyms" in source_specific:
            if not validate_list_of_strings(
                source_specific["synonyms"]
            ):
                errors.append(
                    f"Ligne {line_number}: "
                    f"source_specific.synonyms doit être "
                    f"une liste de chaînes"
                )

        # -----------------------------------------------------
        # choices
        # -----------------------------------------------------
        choices = source_specific.get("choices")

        if not isinstance(choices, dict):
            errors.append(
                f"Ligne {line_number}: "
                f"source_specific.choices doit être un objet"
            )
        else:
            for key, value in choices.items():

                if not isinstance(key, str):
                    errors.append(
                        f"Ligne {line_number}: "
                        f"clé de choices non textuelle"
                    )

                if not isinstance(value, str):
                    errors.append(
                        f"Ligne {line_number}: "
                        f"choice {key!r} doit être une chaîne"
                    )

        # -----------------------------------------------------
        # correct_answers
        # -----------------------------------------------------
        correct_answers = source_specific.get("correct_answers")

        if not validate_list_of_strings(correct_answers):
            errors.append(
                f"Ligne {line_number}: "
                f"source_specific.correct_answers doit être "
                f"une liste de chaînes"
            )

        # -----------------------------------------------------
        # Champs spécifiques MedQuAD
        # -----------------------------------------------------
        dataset = source.get("dataset") if isinstance(source, dict) else None

        if dataset == "MedQuAD":

            if "semantic_types" in source_specific:
                if not validate_list_of_strings(
                    source_specific["semantic_types"]
                ):
                    errors.append(
                        f"Ligne {line_number}: "
                        f"semantic_types doit être une liste de chaînes"
                    )

            if "semantic_group" in source_specific:
                if source_specific["semantic_group"] is not None:
                    if not isinstance(
                        source_specific["semantic_group"],
                        str,
                    ):
                        errors.append(
                            f"Ligne {line_number}: "
                            f"semantic_group doit être une chaîne ou null"
                        )

        # -----------------------------------------------------
        # Champs spécifiques FrenchMedMCQA / MediQAl
        # -----------------------------------------------------
        if dataset in {"FrenchMedMCQA", "MediQAl"}:

            if "medical_subject" in source_specific:
                value = source_specific["medical_subject"]

                if value is not None and not isinstance(value, str):
                    errors.append(
                        f"Ligne {line_number}: "
                        f"medical_subject doit être une chaîne ou null"
                    )

    # =========================================================
    # Quality
    # =========================================================
    quality = record.get("quality")

    if not isinstance(quality, dict):
        errors.append(
            f"Ligne {line_number}: quality n'est pas un objet"
        )
    else:
        missing = EXPECTED_QUALITY - quality.keys()

        if missing:
            errors.append(
                f"Ligne {line_number}: quality champs manquants: "
                f"{sorted(missing)}"
            )

        confidence = quality.get("confidence")

        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                errors.append(
                    f"Ligne {line_number}: "
                    f"confidence doit être un nombre ou null"
                )

        validated = quality.get("validated")

        if not isinstance(validated, bool):
            errors.append(
                f"Ligne {line_number}: "
                f"validated doit être un booléen"
            )

    # =========================================================
    # Provenance
    # =========================================================
    provenance = record.get("provenance")

    if not isinstance(provenance, dict):
        errors.append(
            f"Ligne {line_number}: provenance n'est pas un objet"
        )
    else:
        missing = EXPECTED_PROVENANCE - provenance.keys()

        if missing:
            errors.append(
                f"Ligne {line_number}: provenance champs manquants: "
                f"{sorted(missing)}"
            )

        processing_version = provenance.get(
            "processing_version"
        )

        if not is_non_empty_string(processing_version):
            errors.append(
                f"Ligne {line_number}: "
                f"processing_version absent ou vide"
            )

    return errors


def main():

    print("=" * 60)
    print("VALIDATION DU SCHÉMA SFT - 5 000 ENTRÉES")
    print("=" * 60)
    print()

    if not DATASET_PATH.exists():
        print(
            f"ERREUR : fichier introuvable : "
            f"{DATASET_PATH}"
        )
        return

    errors = []
    seen_ids = set()

    total = 0

    # =========================================================
    # Lecture JSONL
    # =========================================================
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:

                errors.append(
                    f"Ligne {line_number}: "
                    f"JSON invalide: {exc}"
                )

                continue

            record_errors = validate_record(
                record,
                line_number,
                seen_ids,
            )

            errors.extend(record_errors)

    # =========================================================
    # Résultats
    # =========================================================
    print(f"Entrées analysées : {total}")
    print(f"IDs techniques uniques : {len(seen_ids)}")
    print(f"Erreurs : {len(errors)}")
    print()

    if total != 5000:
        print(
            f"⚠️ Nombre d'entrées différent de 5 000 "
            f"(trouvé : {total})"
        )

    if len(seen_ids) != total:
        print(
            "⚠️ Des IDs techniques sont dupliqués"
        )

    if errors:

        print("❌ SCHÉMA NON VALIDÉ")
        print()
        print("Premières erreurs :")

        for error in errors[:50]:
            print(" -", error)

        if len(errors) > 50:
            print()
            print(
                f"... {len(errors) - 50} "
                f"erreurs supplémentaires"
            )

        return

    if total == 5000 and len(seen_ids) == 5000:

        print("✅ SCHÉMA VALIDÉ")
        print("✅ 5 000 entrées valides")
        print("✅ 5 000 IDs techniques uniques")
        print("✅ Tous les champs obligatoires présents")
        print("✅ Types de données corrects")
        print("✅ Valeurs autorisées respectées")
        print("✅ Aucun JSONL invalide")


if __name__ == "__main__":
    main()
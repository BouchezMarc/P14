import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PATH = Path(r"data/processed/sft_5000.jsonl")

EXPECTED_TOTAL = 5000

EXPECTED_SOURCE_COUNTS = {
    "MedQuAD": 1667,
    "FrenchMedMCQA": 1667,
    "MediQAl": 1666,
}

VALID_LANGUAGES = {"fr", "en"}

VALID_TASKS = {
    "medical_qa",
    "medical_mcq",
}

VALID_MCQ_CHOICES = {"A", "B", "C", "D", "E"}

REQUIRED_TOP_LEVEL = {
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

REQUIRED_SOURCE = {
    "dataset",
    "original_id",
    "source",
}

REQUIRED_CLINICAL_CONTEXT = {
    "clinical_case",
    "symptoms",
    "history",
    "vitals",
    "diagnosis",
    "medications",
    "allergies",
}

REQUIRED_SOURCE_SPECIFIC = {
    "question_type",
    "question_focus",
    "synonyms",
    "umls_cui",
    "semantic_types",
    "semantic_group",
    "choices",
    "correct_answers",
}

REQUIRED_QUALITY = {
    "confidence",
    "validated",
}

REQUIRED_PROVENANCE = {
    "source_license",
    "processing_version",
}


# ============================================================
# CHARGEMENT
# ============================================================

if not PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {PATH}")

rows = []
json_errors = []

with PATH.open("r", encoding="utf-8") as f:

    for line_no, line in enumerate(f, 1):

        if not line.strip():
            continue

        try:
            row = json.loads(line)
            rows.append((line_no, row))

        except json.JSONDecodeError as e:
            json_errors.append((line_no, str(e)))


print("=" * 70)
print("CONTROLE SFT")
print("=" * 70)

print(f"\nFichier : {PATH}")
print(f"Entrées chargées : {len(rows)}")


# ============================================================
# ERREURS REELLES
# ============================================================

errors = []


# ============================================================
# 1. NOMBRE TOTAL
# ============================================================

print("\n[1] NOMBRE TOTAL")

if len(rows) == EXPECTED_TOTAL:
    print(f"OK : {len(rows)} / {EXPECTED_TOTAL}")
else:
    print(f"ERREUR : {len(rows)} / {EXPECTED_TOTAL}")
    errors.append(
        f"Nombre total incorrect : {len(rows)}"
    )


# ============================================================
# 2. JSON
# ============================================================

print("\n[2] JSON")

print("Erreurs JSON :", len(json_errors))

if json_errors:
    errors.extend(
        f"Ligne {line}: JSON invalide"
        for line, _ in json_errors
    )


# ============================================================
# 3. STRUCTURE GENERALE
# ============================================================

print("\n[3] STRUCTURE")

missing_top_level = []

for line_no, row in rows:

    missing = REQUIRED_TOP_LEVEL - row.keys()

    if missing:
        missing_top_level.append(
            (line_no, sorted(missing))
        )

print(
    "Entrées avec champs principaux manquants :",
    len(missing_top_level)
)

if missing_top_level:
    errors.extend(
        f"Ligne {line}: champs manquants {fields}"
        for line, fields in missing_top_level
    )


# ============================================================
# 4. INSTRUCTION / RESPONSE
# ============================================================

print("\n[4] INSTRUCTION / RESPONSE")

empty_instruction = []
empty_response = []

for line_no, row in rows:

    if not str(row.get("instruction", "")).strip():
        empty_instruction.append(line_no)

    if not str(row.get("response", "")).strip():
        empty_response.append(line_no)


print("Instructions vides :", len(empty_instruction))
print("Réponses vides     :", len(empty_response))

errors.extend(
    f"Ligne {line}: instruction vide"
    for line in empty_instruction
)

errors.extend(
    f"Ligne {line}: response vide"
    for line in empty_response
)


# ============================================================
# 5. IDS
# ============================================================

print("\n[5] IDS")

ids = []
missing_ids = []

for line_no, row in rows:

    value = row.get("id")

    if not isinstance(value, str) or not value.strip():
        missing_ids.append(line_no)
    else:
        ids.append(value)


duplicate_ids = {
    value: count
    for value, count in Counter(ids).items()
    if count > 1
}

print("IDs vides     :", len(missing_ids))
print("IDs dupliqués :", len(duplicate_ids))

errors.extend(
    f"Ligne {line}: ID absent"
    for line in missing_ids
)

errors.extend(
    f"ID dupliqué : {value}"
    for value in duplicate_ids
)


# ============================================================
# 6. SOURCE
# ============================================================

print("\n[6] SOURCES")

source_counts = Counter(
    row.get("source", {}).get("dataset")
    for _, row in rows
)

for source, count in source_counts.items():
    print(f"  {source}: {count}")

print("\nComparaison attendue :")

for source, expected in EXPECTED_SOURCE_COUNTS.items():

    actual = source_counts.get(source, 0)

    status = "OK" if actual == expected else "ERREUR"

    print(
        f"  {source}: {actual} / {expected} {status}"
    )

    if actual != expected:
        errors.append(
            f"Source {source}: {actual} au lieu de {expected}"
        )


# ============================================================
# 7. TASK
# ============================================================

print("\n[7] TASK")

task_counts = Counter(
    row.get("task")
    for _, row in rows
)

for task, count in task_counts.items():
    print(f"  {task}: {count}")

invalid_tasks = [
    (line, row.get("task"))
    for line, row in rows
    if row.get("task") not in VALID_TASKS
]

print(
    "Tasks invalides :",
    len(invalid_tasks)
)

errors.extend(
    f"Ligne {line}: task invalide {task}"
    for line, task in invalid_tasks
)


# ============================================================
# 8. LANGUE FR / EN
# ============================================================

print("\n[8] LANGUE FR / EN")

language_counts = Counter(
    str(row.get("language", "")).lower()
    for _, row in rows
)

for language, count in language_counts.items():
    print(f"  {language.upper()}: {count}")

invalid_languages = [
    (line, row.get("language"))
    for line, row in rows
    if str(row.get("language", "")).lower()
    not in VALID_LANGUAGES
]

print(
    "Langues invalides :",
    len(invalid_languages)
)

errors.extend(
    f"Ligne {line}: langue invalide {language}"
    for line, language in invalid_languages
)


# ============================================================
# 9. SOURCE <-> LANGUE
# ============================================================

print("\n[9] COHERENCE SOURCE / LANGUE")

source_language_errors = []

for line, row in rows:

    source = row.get("source", {}).get("dataset")
    language = str(
        row.get("language", "")
    ).lower()

    if source == "MedQuAD" and language != "en":

        source_language_errors.append(
            (line, source, language)
        )

    elif source == "FrenchMedMCQA" and language != "fr":

        source_language_errors.append(
            (line, source, language)
        )


print(
    "Incohérences source/langue :",
    len(source_language_errors)
)

errors.extend(
    f"Ligne {line}: {source} / {language}"
    for line, source, language in source_language_errors
)

# ============================================================
# 10. DOUBLONS
# ============================================================

import re
from collections import defaultdict


def normalize_text(text):
    """Normalisation légère pour comparer les textes."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# ------------------------------------------------------------
# 10.1 Questions répétées
# ------------------------------------------------------------

question_groups = defaultdict(list)

for line_no, row in rows:
    instruction = normalize_text(row.get("instruction"))
    question_groups[instruction].append((line_no, row))


repeated_question_groups = {
    question: entries
    for question, entries in question_groups.items()
    if len(entries) > 1
}


# ------------------------------------------------------------
# 10.2 Vrais doublons : même question + même réponse
# ------------------------------------------------------------

question_response_groups = defaultdict(list)

for line_no, row in rows:
    instruction = normalize_text(row.get("instruction"))
    response = normalize_text(row.get("response"))

    key = (instruction, response)

    question_response_groups[key].append((line_no, row))


exact_duplicate_groups = {
    key: entries
    for key, entries in question_response_groups.items()
    if len(entries) > 1
}


print("\nDUPLICATS")
print("=" * 80)

print(
    f"Questions répétées : {len(repeated_question_groups)} groupes "
    f"(informatif)"
)

print(
    f"Doublons exacts question + réponse : "
    f"{len(exact_duplicate_groups)} groupes"
)


if exact_duplicate_groups:
    print("\n" + "-" * 80)
    print("DOUBLONS EXACTS")
    print("-" * 80)

    for i, ((question, response), entries) in enumerate(
        exact_duplicate_groups.items(), 1
    ):
        print(f"\nDUPLICAT EXACT #{i}")
        print(f"QUESTION : {question}")

        for line_no, row in entries:
            source = row.get("source", {})
            print(f"\n  Ligne       : {line_no}")
            print(f"  ID          : {row.get('id')}")
            print(f"  Source      : {source.get('dataset')}")
            print(f"  Original ID : {source.get('original_id')}")
            print(f"  Task        : {row.get('task')}")
            print(f"  Langue      : {row.get('language')}")


# ============================================================
# 16. INSTRUCTIONS TRES COURTES
# ============================================================

very_short_instructions = []

for line_no, row in rows:
    instruction = str(row.get("instruction", "")).strip()
    task = row.get("task")

    # Les questions très courtes sont normales pour les MCQ.
    # On ne les contrôle donc que pour medical_qa.
    if task == "medical_qa" and len(instruction) < 10:
        very_short_instructions.append(
            (line_no, row)
        )


print("\n" + "=" * 80)
print("INSTRUCTIONS TRES COURTES")
print("=" * 80)

print(
    f"Instructions < 10 caractères pour medical_qa : "
    f"{len(very_short_instructions)}"
)

for line_no, row in very_short_instructions:
    print("\n" + "-" * 80)

    source = row.get("source", {})

    print(f"Ligne       : {line_no}")
    print(f"ID          : {row.get('id')}")
    print(f"Source      : {source.get('dataset')}")
    print(f"Task        : {row.get('task')}")
    print(f"Instruction : {row.get('instruction')!r}")
    print(f"Response    : {row.get('response')}")

# ============================================================
# 11. CLINICAL CONTEXT
# ============================================================

print("\n[11] CLINICAL CONTEXT")

clinical_context_errors = []

for line, row in rows:

    context = row.get("clinical_context")

    if not isinstance(context, dict):

        clinical_context_errors.append(
            (line, "clinical_context non-dict")
        )

        continue

    missing = (
        REQUIRED_CLINICAL_CONTEXT
        - context.keys()
    )

    if missing:

        clinical_context_errors.append(
            (line, f"champs manquants: {sorted(missing)}")
        )

    # Les champs listés doivent être des listes
    for field in [
        "symptoms",
        "history",
        "vitals",
        "medications",
        "allergies",
    ]:

        if field in context and not isinstance(
            context[field], list
        ):

            clinical_context_errors.append(
                (line, f"{field} doit être une liste")
            )


print(
    "Erreurs clinical_context :",
    len(clinical_context_errors)
)

errors.extend(
    f"Ligne {line}: {problem}"
    for line, problem in clinical_context_errors
)


# ============================================================
# 12. SOURCE SPECIFIC
# ============================================================

print("\n[12] SOURCE_SPECIFIC")

source_specific_errors = []

for line, row in rows:

    data = row.get("source_specific")

    if not isinstance(data, dict):

        source_specific_errors.append(
            (line, "source_specific non-dict")
        )

        continue

    source = row.get(
        "source",
        {}
    ).get("dataset")

    task = row.get("task")

    # --------------------------------------------------------
    # Champs communs réellement obligatoires
    # --------------------------------------------------------

    common_required = {
        "choices",
        "correct_answers",
    }

    missing = common_required - data.keys()

    if missing:

        source_specific_errors.append(
            (
                line,
                f"champs communs manquants: {sorted(missing)}"
            )
        )

    # --------------------------------------------------------
    # MedQuAD
    # --------------------------------------------------------

    if source == "MedQuAD":

        required = {
            "question_type",
            "question_focus",
            "synonyms",
            "umls_cui",
            "semantic_types",
            "semantic_group",
            "choices",
            "correct_answers",
        }

        missing = required - data.keys()

        if missing:

            source_specific_errors.append(
                (
                    line,
                    f"MedQuAD champs manquants: {sorted(missing)}"
                )
            )

    # --------------------------------------------------------
    # FrenchMedMCQA
    # --------------------------------------------------------

    elif source == "FrenchMedMCQA":

        # Pour le moment, on vérifie uniquement
        # la structure réellement commune aux MCQ.
        pass

    # --------------------------------------------------------
    # MediQAl
    # --------------------------------------------------------

    elif source == "MediQAl":

        # Pour le moment, on vérifie uniquement
        # la structure réellement commune aux MCQ.
        pass

    # --------------------------------------------------------
    # Source inconnue
    # --------------------------------------------------------

    else:

        source_specific_errors.append(
            (
                line,
                f"source inconnue: {source}"
            )
        )


print(
    "Erreurs structurelles :",
    len(source_specific_errors)
)

errors.extend(
    f"Ligne {line}: {problem}"
    for line, problem in source_specific_errors
)


# ============================================================
# 13. MCQ
# ============================================================

print("\n[13] MCQ")

mcq_rows = [
    (line, row)
    for line, row in rows
    if row.get("task") == "medical_mcq"
]

print(
    "Nombre total MCQ :",
    len(mcq_rows)
)

bad_choices = []
bad_correct = []
empty_choices = []

for line, row in mcq_rows:

    data = row.get(
        "source_specific",
        {}
    )

    choices = data.get(
        "choices",
        {}
    )

    correct_answers = data.get(
        "correct_answers",
        []
    )

    if set(choices.keys()) != VALID_MCQ_CHOICES:

        bad_choices.append(line)

    for choice in VALID_MCQ_CHOICES:

        if not str(
            choices.get(choice, "")
        ).strip():

            empty_choices.append(
                (line, choice)
            )

    if (
        not isinstance(correct_answers, list)
        or not correct_answers
        or any(
            answer not in VALID_MCQ_CHOICES
            for answer in correct_answers
        )
        or any(
            answer not in choices
            or not str(
                choices.get(answer, "")
            ).strip()
            for answer in correct_answers
        )
    ):

        bad_correct.append(line)


print(
    "Choix A-E incorrects :",
    len(bad_choices)
)

print(
    "Réponses correctes incorrectes :",
    len(bad_correct)
)

print(
    "Choix vides :",
    len(empty_choices)
)

errors.extend(
    f"Ligne {line}: structure MCQ invalide"
    for line in bad_choices
)

errors.extend(
    f"Ligne {line}: correct_answers invalide"
    for line in bad_correct
)

errors.extend(
    f"Ligne {line}: choix {choice} vide"
    for line, choice in empty_choices
)


# ============================================================
# 14. QUALITY
# ============================================================

print("\n[14] QUALITY")

quality_errors = []

for line, row in rows:

    quality = row.get("quality")

    if not isinstance(quality, dict):

        quality_errors.append(
            (line, "quality non-dict")
        )

        continue

    missing = REQUIRED_QUALITY - quality.keys()

    if missing:

        quality_errors.append(
            (line, f"champs manquants: {sorted(missing)}")
        )

        continue

    confidence = quality.get(
        "confidence"
    )

    validated = quality.get(
        "validated"
    )

    # confidence peut être null
    if confidence is not None:

        if not isinstance(
            confidence,
            (int, float)
        ):

            quality_errors.append(
                (line, "confidence non numérique")
            )

        elif not 0 <= confidence <= 1:

            quality_errors.append(
                (line, "confidence hors [0,1]")
            )

    # validated doit être booléen
    if not isinstance(
        validated,
        bool
    ):

        quality_errors.append(
            (line, "validated doit être booléen")
        )


print(
    "Erreurs quality :",
    len(quality_errors)
)

errors.extend(
    f"Ligne {line}: {problem}"
    for line, problem in quality_errors
)


# ============================================================
# 15. PROVENANCE
# ============================================================

print("\n[15] PROVENANCE")

provenance_errors = []

for line, row in rows:

    provenance = row.get("provenance")

    if not isinstance(provenance, dict):

        provenance_errors.append(
            (line, "provenance non-dict")
        )

        continue

    missing = (
        REQUIRED_PROVENANCE
        - provenance.keys()
    )

    if missing:

        provenance_errors.append(
            (line, f"champs manquants: {sorted(missing)}")
        )


print(
    "Erreurs provenance :",
    len(provenance_errors)
)

errors.extend(
    f"Ligne {line}: {problem}"
    for line, problem in provenance_errors
)


# ============================================================
# 16. PII BASIQUE
# ============================================================

print("\n[16] DETECTION PII BASIQUE")

PII_PATTERNS = {

    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    "telephone": re.compile(
        r"(?<!\d)"
        r"(?:\+33|0)[1-9]"
        r"(?:[\s.-]?\d{2}){4}"
        r"(?!\d)"
    ),

    "date": re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])"
        r"[/-]"
        r"(?:0?[1-9]|1[0-2])"
        r"[/-]"
        r"(?:19|20)\d{2}\b"
    ),
}

pii_hits = []

for line, row in rows:

    text = (
        str(row.get("instruction", ""))
        + " "
        + str(row.get("response", ""))
    )

    for pii_type, pattern in PII_PATTERNS.items():

        if pattern.search(text):

            pii_hits.append(
                (line, pii_type)
            )


pii_counts = Counter(
    pii_type
    for _, pii_type in pii_hits
)

for pii_type, count in pii_counts.items():

    print(
        f"  {pii_type}: {count}"
    )

print(
    "Entrées avec motif PII :",
    len(set(line for line, _ in pii_hits))
)

print(
    "INFO : les motifs PII sont signalés mais ne sont pas comptés comme erreurs automatiques."
)


# ============================================================
# 17. TEXTES TRES COURTS
# ============================================================

print("\n[17] CONTENU TEXTUEL")

very_short_instruction = []

very_short_response = []

for line, row in rows:

    instruction = str(
        row.get("instruction", "")
    ).strip()

    response = str(
        row.get("response", "")
    ).strip()

    if len(instruction) < 10:

        very_short_instruction.append(
            (line, instruction)
        )

    if len(response) < 10:

        very_short_response.append(
            (line, response)
        )


print(
    "Instructions < 10 caractères :",
    len(very_short_instruction)
)

print(
    "Réponses < 10 caractères :",
    len(very_short_response)
)

print(
    "INFO : signalement uniquement, pas une erreur automatique."
)


# ============================================================
# RESULTAT FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESULTAT FINAL")
print("=" * 70)

print(
    f"Erreurs techniques réelles : {len(errors)}"
)

if errors:

    print("\n❌ CONTROLE TECHNIQUE : ERREURS")

    print("\nPremières erreurs :")

    for error in errors[:20]:

        print(
            f"  - {error}"
        )

else:

    print(
        "✅ CONTROLE TECHNIQUE : OK"
    )


print("\nRésumé :")

print(
    f"  Total                    : {len(rows)}"
)

print(
    f"  Sources                  : {len(source_counts)}"
)

print(
    f"  Langues                  : {dict(language_counts)}"
)

print(
    f"  MCQ                      : {len(mcq_rows)}"
)

print(
    f"  Instructions vides       : {len(empty_instruction)}"
)

print(
    f"  Réponses vides           : {len(empty_response)}"
)

print(
    f"  IDs dupliqués            : {len(duplicate_ids)}"
)

print(
    f"  Questions répétées       : {len(repeated_question_groups)}"
    f"\n  Doublons exacts Q+R      : {len(exact_duplicate_groups)}"
)

print(
    f"  Erreurs source/langue    : {len(source_language_errors)}"
)

print(
    f"  Erreurs clinical_context : {len(clinical_context_errors)}"
)

print(
    f"  Erreurs source_specific  : {len(source_specific_errors)}"
)

print(
    f"  Erreurs MCQ              : "
    f"{len(bad_choices) + len(bad_correct) + len(empty_choices)}"
)

print(
    f"  Erreurs quality          : {len(quality_errors)}"
)

print(
    f"  Erreurs provenance       : {len(provenance_errors)}"
)

print(
    f"  PII à vérifier           : "
    f"{len(set(line for line, _ in pii_hits))}"
)

print(
    f"  Instructions très courtes: "
    f"{len(very_short_instruction)}"
)
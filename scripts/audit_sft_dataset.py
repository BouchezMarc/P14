import json
import os
import statistics
from collections import Counter, defaultdict

from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "data/processed/sft_train.jsonl"
OUTPUT_FILE = "results/sft_dataset_audit.txt"

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
MAX_SEQ_LENGTH = 2048

N_LONGEST = 20
N_SHORTEST = 10
N_SAMPLES_PER_SOURCE = 3


# ============================================================
# UTILITAIRES
# ============================================================

def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values))

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def describe(values):
    if not values:
        return {
            "min": 0,
            "p25": 0,
            "median": 0,
            "mean": 0,
            "p75": 0,
            "p95": 0,
            "max": 0,
        }

    return {
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def print_stats(name, values, out):
    s = describe(values)

    out.append(f"{name}")
    out.append("-" * 70)
    out.append(f"min     : {s['min']:.1f}")
    out.append(f"p25     : {s['p25']:.1f}")
    out.append(f"median  : {s['median']:.1f}")
    out.append(f"mean    : {s['mean']:.1f}")
    out.append(f"p75     : {s['p75']:.1f}")
    out.append(f"p95     : {s['p95']:.1f}")
    out.append(f"max     : {s['max']:.1f}")
    out.append("")


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 70)
print("AUDIT DU DATASET SFT")
print("=" * 70)
print()
print(f"Dataset : {DATASET_FILE}")
print(f"Tokenizer : {MODEL_NAME}")
print(f"Max sequence length : {MAX_SEQ_LENGTH}")
print()

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset introuvable : {DATASET_FILE}"
    )

records = []

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON invalide ligne {line_number}: {e}"
            )

        record["_line_number"] = line_number
        records.append(record)


# ============================================================
# TOKENIZER
# ============================================================

print("Chargement du tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Tokenizer chargé.")
print()


# ============================================================
# STRUCTURES
# ============================================================

languages = Counter()
tasks = Counter()
sources = Counter()

instructions_lengths = []
responses_lengths = []
full_lengths = []

over_2048 = []
over_2048_response_only = []

records_by_source = defaultdict(list)

ids = Counter()

missing_fields = Counter()

duplicate_records = []

formatted_lengths = []


# ============================================================
# ANALYSE
# ============================================================

for record in records:

    record_id = record.get("id")

    if record_id is not None:
        ids[record_id] += 1

    language = record.get("language", "MISSING")
    task = record.get("task", "MISSING")

    languages[language] += 1
    tasks[task] += 1

    source_info = record.get("source", {})

    if isinstance(source_info, dict):
        source = source_info.get(
            "dataset",
            source_info.get("source", "UNKNOWN")
        )
    else:
        source = str(source_info)

    sources[source] += 1
    records_by_source[source].append(record)

    instruction = record.get("instruction")
    response = record.get("response")

    if instruction is None:
        missing_fields["instruction"] += 1
        instruction = ""

    if response is None:
        missing_fields["response"] += 1
        response = ""

    # --------------------------------------------------------
    # TOKENISATION
    # --------------------------------------------------------

    instruction_tokens = tokenizer(
        instruction,
        add_special_tokens=False,
    )["input_ids"]

    response_tokens = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    formatted_text = (
        "You are a medical question-answering assistant.\n\n"
        "Respond in the same language as the question.\n"
        "Answer only the question asked.\n"
        "Provide a clear, factual and clinically relevant answer.\n"
        "Do not introduce unrelated diseases, conditions, or topics.\n"
        "Do not invent medical facts, diagnoses, treatments, or recommendations.\n"
        "When the available information is insufficient, state the uncertainty clearly.\n"
        "Do not provide dangerous or unsupported medical recommendations.\n\n"
        "Question:\n"
        f"{instruction}\n\n"
        "Answer:\n"
        f"{response}"
    )

    full_tokens = tokenizer(
        formatted_text,
        add_special_tokens=True,
    )["input_ids"]

    instruction_len = len(instruction_tokens)
    response_len = len(response_tokens)
    full_len = len(full_tokens)

    instructions_lengths.append(instruction_len)
    responses_lengths.append(response_len)
    full_lengths.append(full_len)
    formatted_lengths.append(full_len)

    if full_len > MAX_SEQ_LENGTH:
        over_2048.append(record)

    if response_len > MAX_SEQ_LENGTH:
        over_2048_response_only.append(record)


# ============================================================
# DOUBLONS
# ============================================================

duplicate_ids = {
    record_id: count
    for record_id, count in ids.items()
    if count > 1
}


# ============================================================
# RAPPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("AUDIT DU DATASET SFT")
report.append("=" * 70)
report.append("")

report.append(f"Fichier              : {DATASET_FILE}")
report.append(f"Nombre de records    : {len(records)}")
report.append(f"Tokenizer             : {MODEL_NAME}")
report.append(f"MAX_SEQ_LENGTH        : {MAX_SEQ_LENGTH}")
report.append("")

# ------------------------------------------------------------
# LANGUES
# ------------------------------------------------------------

report.append("=" * 70)
report.append("1. LANGUES")
report.append("=" * 70)

for language, count in languages.most_common():
    percentage = count / len(records) * 100
    report.append(
        f"{language:15} : {count:6} ({percentage:6.2f} %)"
    )

report.append("")

# ------------------------------------------------------------
# TASKS
# ------------------------------------------------------------

report.append("=" * 70)
report.append("2. TASKS")
report.append("=" * 70)

for task, count in tasks.most_common():
    percentage = count / len(records) * 100
    report.append(
        f"{task:30} : {count:6} ({percentage:6.2f} %)"
    )

report.append("")

# ------------------------------------------------------------
# SOURCES
# ------------------------------------------------------------

report.append("=" * 70)
report.append("3. SOURCES")
report.append("=" * 70)

for source, count in sources.most_common():
    percentage = count / len(records) * 100
    report.append(
        f"{source:30} : {count:6} ({percentage:6.2f} %)"
    )

report.append("")

# ------------------------------------------------------------
# LONGUEURS
# ------------------------------------------------------------

report.append("=" * 70)
report.append("4. LONGUEURS TOKEN")
report.append("=" * 70)
report.append("")

print_stats(
    "INSTRUCTIONS",
    instructions_lengths,
    report,
)

print_stats(
    "RESPONSES",
    responses_lengths,
    report,
)

print_stats(
    "TEXTE COMPLET DU SFT",
    full_lengths,
    report,
)

# ------------------------------------------------------------
# TRONCATURE
# ------------------------------------------------------------

report.append("=" * 70)
report.append("5. RISQUE DE TRONCATURE À 2048 TOKENS")
report.append("=" * 70)
report.append("")

report.append(
    f"Exemples > {MAX_SEQ_LENGTH} tokens : "
    f"{len(over_2048)} "
    f"({len(over_2048) / len(records) * 100:.2f} %)"
)

report.append(
    f"Réponses seules > {MAX_SEQ_LENGTH} tokens : "
    f"{len(over_2048_response_only)} "
    f"({len(over_2048_response_only) / len(records) * 100:.2f} %)"
)

report.append("")

# ------------------------------------------------------------
# DOUBLONS
# ------------------------------------------------------------

report.append("=" * 70)
report.append("6. DOUBLONS")
report.append("=" * 70)
report.append("")

report.append(
    f"IDs uniques        : {len(ids)}"
)

report.append(
    f"IDs dupliqués      : {len(duplicate_ids)}"
)

if duplicate_ids:
    report.append("")
    report.append("IDs concernés :")

    for record_id, count in list(
        duplicate_ids.items()
    )[:50]:
        report.append(
            f"  {record_id} : {count} occurrences"
        )

report.append("")

# ------------------------------------------------------------
# CHAMPS MANQUANTS
# ------------------------------------------------------------

report.append("=" * 70)
report.append("7. CHAMPS MANQUANTS")
report.append("=" * 70)
report.append("")

if missing_fields:
    for field, count in missing_fields.items():
        report.append(
            f"{field:20} : {count}"
        )
else:
    report.append("Aucun champ manquant détecté.")

report.append("")

# ------------------------------------------------------------
# PLUS LONGUES REPONSES
# ------------------------------------------------------------

longest = sorted(
    records,
    key=lambda r: len(
        tokenizer(
            r.get("response", ""),
            add_special_tokens=False,
        )["input_ids"]
    ),
    reverse=True,
)

report.append("=" * 70)
report.append(
    f"8. {N_LONGEST} RÉPONSES LES PLUS LONGUES"
)
report.append("=" * 70)
report.append("")

for i, record in enumerate(longest[:N_LONGEST], start=1):

    response = record.get("response", "")

    response_tokens = len(
        tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]
    )

    report.append(
        f"[{i:02d}] "
        f"id={record.get('id')} "
        f"lang={record.get('language')} "
        f"tokens={response_tokens}"
    )

    report.append(
        f"QUESTION: {record.get('instruction', '')[:300]}"
    )

    report.append(
        f"RESPONSE: {response[:700]}"
    )

    report.append("-" * 70)

# ------------------------------------------------------------
# EXEMPLES PAR SOURCE
# ------------------------------------------------------------

report.append("=" * 70)
report.append("9. EXEMPLES PAR SOURCE")
report.append("=" * 70)
report.append("")

for source, source_records in records_by_source.items():

    report.append("")
    report.append(f"SOURCE : {source}")
    report.append("-" * 70)

    for record in source_records[:N_SAMPLES_PER_SOURCE]:

        report.append(
            f"ID       : {record.get('id')}"
        )

        report.append(
            f"LANGUAGE : {record.get('language')}"
        )

        report.append(
            f"QUESTION : {record.get('instruction', '')[:500]}"
        )

        report.append(
            f"RESPONSE : {record.get('response', '')[:1000]}"
        )

        report.append("-" * 70)

# ------------------------------------------------------------
# EXEMPLES DÉPASSANT 2048
# ------------------------------------------------------------

report.append("=" * 70)
report.append(
    f"10. EXEMPLES DÉPASSANT {MAX_SEQ_LENGTH} TOKENS"
)
report.append("=" * 70)
report.append("")

for record in over_2048[:20]:

    full_text = (
        "You are a medical question-answering assistant.\n\n"
        "Respond in the same language as the question.\n"
        "Answer only the question asked.\n"
        "Provide a clear, factual and clinically relevant answer.\n"
        "Do not introduce unrelated diseases, conditions, or topics.\n"
        "Do not invent medical facts, diagnoses, treatments, or recommendations.\n"
        "When the available information is insufficient, state the uncertainty clearly.\n"
        "Do not provide dangerous or unsupported medical recommendations.\n\n"
        "Question:\n"
        f"{record.get('instruction', '')}\n\n"
        "Answer:\n"
        f"{record.get('response', '')}"
    )

    token_count = len(
        tokenizer(
            full_text,
            add_special_tokens=True,
        )["input_ids"]
    )

    report.append(
        f"ID={record.get('id')} "
        f"tokens={token_count} "
        f"language={record.get('language')}"
    )

    report.append(
        f"QUESTION: {record.get('instruction', '')[:500]}"
    )

    report.append(
        f"RESPONSE: {record.get('response', '')[:1500]}"
    )

    report.append("-" * 70)

# ------------------------------------------------------------
# CONCLUSION AUTOMATIQUE
# ------------------------------------------------------------

report.append("=" * 70)
report.append("11. INDICATEURS À SURVEILLER")
report.append("=" * 70)
report.append("")

truncation_rate = (
    len(over_2048) / len(records) * 100
    if records else 0
)

very_long_rate = (
    sum(x > 1000 for x in responses_lengths)
    / len(records) * 100
    if records else 0
)

report.append(
    f"Réponses > 1000 tokens : "
    f"{sum(x > 1000 for x in responses_lengths)} "
    f"({very_long_rate:.2f} %)"
)

report.append(
    f"Exemples > 2048 tokens : "
    f"{len(over_2048)} "
    f"({truncation_rate:.2f} %)"
)

report.append("")

report.append(
    "IMPORTANT : cet audit ne modifie pas le dataset."
)

report.append(
    "Il sert uniquement à diagnostiquer le pipeline SFT."
)

# ============================================================
# SAUVEGARDE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(report))


# ============================================================
# AFFICHAGE
# ============================================================

print()
print("=" * 70)
print("AUDIT TERMINÉ")
print("=" * 70)
print()
print(f"Records analysés : {len(records)}")
print(f"FR / EN          : {dict(languages)}")
print(f"Sources          : {dict(sources)}")
print()
print(
    f"Exemples > 2048 tokens : "
    f"{len(over_2048)} "
    f"({truncation_rate:.2f} %)"
)
print(
    f"Réponses > 1000 tokens : "
    f"{sum(x > 1000 for x in responses_lengths)} "
    f"({very_long_rate:.2f} %)"
)
print()
print(f"Rapport : {OUTPUT_FILE}")
print("=" * 70)
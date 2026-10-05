import json
import re
from collections import Counter, defaultdict

TEST_FILE = "data/processed/sft_test.jsonl"
BASE_EVAL_FILE = "results/base/predictions.jsonl"
SFT_EVAL_FILE = "results/sft_v5/predictions.jsonl"
#SFT_EVAL_FILE = "results/dpo_v2/predictions.jsonl"

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def get_task(record):
    return record.get("task")


def get_language(record):
    return record.get("language")


def get_source(record):
    source = record.get("source", {})

    if isinstance(source, dict):
        return source.get("dataset", "UNKNOWN")

    return source or "UNKNOWN"


def get_question_type(record):
    source_specific = record.get("source_specific", {})

    if isinstance(source_specific, dict):
        return source_specific.get("question_type", "UNKNOWN")

    return "UNKNOWN"


def get_correct_answers(record):
    source_specific = record.get("source_specific", {})

    if not isinstance(source_specific, dict):
        return []

    answers = source_specific.get("correct_answers", [])

    if not isinstance(answers, list):
        return []

    return sorted(
        {
            str(answer).strip().upper()
            for answer in answers
            if str(answer).strip().upper() in {"A", "B", "C", "D", "E"}
        }
    )


def normalize_prediction(text):
    """
    Extrait les lettres A-E de la partie "Réponses correctes".

    Exemple :
        Réponses correctes :
        B. ...
        C. ...
        D. ...

    => ["B", "C", "D"]
    """

    if not text:
        return []

    text = str(text)

    # Cherche explicitement la section de réponse.
    match = re.search(
        r"Réponses?\s+correctes?\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        answer_section = match.group(1)
    else:
        # Variante anglaise éventuelle
        match = re.search(
            r"Correct\s+answers?\s*:\s*(.*)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            answer_section = match.group(1)
        else:
            answer_section = text

    # On cherche les lettres qui introduisent explicitement une option :
    # B.
    # C)
    # D -
    # etc.
    matches = re.findall(
        r"(?:^|\n)\s*([A-E])\s*[\.\):\-]",
        answer_section,
        flags=re.IGNORECASE,
    )

    if not matches:
        # Cas où plusieurs réponses sont sur la même ligne :
        # "B. ... C. ... D. ..."
        matches = re.findall(
            r"\b([A-E])\s*[\.\):\-]",
            answer_section,
            flags=re.IGNORECASE,
        )

    return sorted(set(letter.upper() for letter in matches))


def compare_sets(predicted, reference):
    predicted = set(predicted)
    reference = set(reference)

    if predicted == reference:
        category = "EXACT"
    elif predicted < reference:
        category = "UNDER"
    elif predicted > reference:
        category = "OVER"
    else:
        category = "WRONG"

    intersection = predicted & reference

    precision = (
        len(intersection) / len(predicted)
        if predicted
        else 0.0
    )

    recall = (
        len(intersection) / len(reference)
        if reference
        else 0.0
    )

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return category, precision, recall, f1


def get_prediction(record):
    """
    Supporte plusieurs structures possibles des fichiers d'évaluation.
    """

    for key in ("prediction", "response", "generated_text", "answer"):
        value = record.get(key)

        if isinstance(value, str):
            return value

    return ""


def build_eval_index(records):
    index = {}

    for record in records:
        record_id = record.get("id")

        if record_id is not None:
            index[str(record_id)] = record

    return index


def evaluate_model(test_records, eval_records):
    eval_index = build_eval_index(eval_records)

    results = []

    for test in test_records:

        if get_task(test) != "medical_mcq":
            continue

        test_id = str(test.get("id"))

        evaluation = eval_index.get(test_id)

        if evaluation is None:
            continue

        reference = get_correct_answers(test)

        prediction_text = get_prediction(evaluation)
        prediction = normalize_prediction(prediction_text)

        category, precision, recall, f1 = compare_sets(
            prediction,
            reference,
        )

        results.append(
            {
                "id": test_id,
                "source": get_source(test),
                "question_type": get_question_type(test),
                "language": get_language(test),
                "reference": reference,
                "prediction": prediction,
                "category": category,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return results


def print_summary(name, results):

    n = len(results)

    if n == 0:
        print(f"\n{name}")
        print("Aucun résultat.")
        return

    exact = sum(r["category"] == "EXACT" for r in results)
    under = sum(r["category"] == "UNDER" for r in results)
    over = sum(r["category"] == "OVER" for r in results)
    wrong = sum(r["category"] == "WRONG" for r in results)

    precision = sum(r["precision"] for r in results) / n
    recall = sum(r["recall"] for r in results) / n
    f1 = sum(r["f1"] for r in results) / n

    print(f"\n{name}")
    print("=" * 70)
    print(f"Examples   : {n}")
    print(f"Exact      : {exact / n:.4f} ({exact}/{n})")
    print(f"Under      : {under / n:.4f} ({under}/{n})")
    print(f"Over       : {over / n:.4f} ({over}/{n})")
    print(f"Wrong      : {wrong / n:.4f} ({wrong}/{n})")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1         : {f1:.4f}")


def print_breakdown(name, results, field):

    groups = defaultdict(list)

    for result in results:
        groups[result[field]].append(result)

    print(f"\n{name}")
    print("=" * 100)

    print(
        f"{'GROUP':<25}"
        f"{'N':>6}"
        f"{'EXACT':>10}"
        f"{'UNDER':>10}"
        f"{'OVER':>10}"
        f"{'WRONG':>10}"
        f"{'P':>10}"
        f"{'R':>10}"
        f"{'F1':>10}"
    )

    for group, rows in sorted(groups.items(), key=lambda x: str(x[0])):

        n = len(rows)

        exact = sum(r["category"] == "EXACT" for r in rows)
        under = sum(r["category"] == "UNDER" for r in rows)
        over = sum(r["category"] == "OVER" for r in rows)
        wrong = sum(r["category"] == "WRONG" for r in rows)

        p = sum(r["precision"] for r in rows) / n
        r = sum(r["recall"] for r in rows) / n
        f1 = sum(r["f1"] for r in rows) / n

        print(
            f"{str(group):<25}"
            f"{n:>6}"
            f"{exact / n:>9.1%}"
            f"{under / n:>9.1%}"
            f"{over / n:>9.1%}"
            f"{wrong / n:>9.1%}"
            f"{p:>9.1%}"
            f"{r:>9.1%}"
            f"{f1:>9.1%}"
        )


def print_answer_count_distribution(test_records):

    counts = Counter()

    for record in test_records:

        if get_task(record) != "medical_mcq":
            continue

        answers = get_correct_answers(record)
        counts[len(answers)] += 1

    print("\nNOMBRE DE BONNES RÉPONSES — QCM")
    print("=" * 50)

    total = sum(counts.values())

    for count, n in sorted(counts.items()):
        print(
            f"{count}: {n} "
            f"({n / total:.2%})"
        )


def print_parser_control(test_records, limit=20):

    """
    Contrôle manuel limité aux exemples MULTI.
    Permet de vérifier que le parseur récupère bien
    B,C,D et non seulement B.
    """

    print("\nCONTRÔLE DU PARSEUR — 20 QCM MULTI")
    print("=" * 80)

    shown = 0

    for record in test_records:

        if get_task(record) != "medical_mcq":
            continue

        reference = get_correct_answers(record)

        if len(reference) <= 1:
            continue

        predicted = normalize_prediction(record.get("response", ""))

        print(f"\nID         : {record.get('id')}")
        print(f"Source     : {get_source(record)}")
        print(f"Référence  : {reference}")
        print(f"Prédiction : {predicted}")

        shown += 1

        if shown >= limit:
            break


# ============================================================
# MAIN
# ============================================================

test_records = load_jsonl(TEST_FILE)

base_records = load_jsonl(BASE_EVAL_FILE)
sft_records = load_jsonl(SFT_EVAL_FILE)


# ------------------------------------------------------------
# Contrôle du dataset
# ------------------------------------------------------------

qcm_records = [
    record
    for record in test_records
    if get_task(record) == "medical_mcq"
]

print("\nDATASET TEST")
print("=" * 70)
print(f"Total test       : {len(test_records)}")
print(f"QCM              : {len(qcm_records)}")
print(f"QA               : {len(test_records) - len(qcm_records)}")


print_answer_count_distribution(test_records)

print_parser_control(test_records)


# ------------------------------------------------------------
# Évaluation
# ------------------------------------------------------------

base_results = evaluate_model(
    test_records,
    base_records,
)

sft_results = evaluate_model(
    test_records,
    sft_records,
)


print_summary("BASE — QCM", base_results)
print_summary("SFT v4 — QCM", sft_results)


print_breakdown(
    "BASE — PAR SOURCE",
    base_results,
    "source",
)

print_breakdown(
    "SFT v4 — PAR SOURCE",
    sft_results,
    "source",
)


print_breakdown(
    "BASE — PAR QUESTION TYPE",
    base_results,
    "question_type",
)

print_breakdown(
    "SFT v4 — PAR QUESTION TYPE",
    sft_results,
    "question_type",
)


# ------------------------------------------------------------
# SINGLE vs MULTI
# ------------------------------------------------------------

def add_answer_cardinality(results):
    for result in results:
        result["cardinality"] = (
            "SINGLE"
            if len(result["reference"]) == 1
            else "MULTI"
        )


add_answer_cardinality(base_results)
add_answer_cardinality(sft_results)

from collections import Counter


def print_prediction_cardinality(name, results):

    reference_counts = Counter(len(r["reference"]) for r in results)
    prediction_counts = Counter(len(r["prediction"]) for r in results)

    print(f"\n{name}")
    print("=" * 70)

    print("\nRÉFÉRENCE — nombre de bonnes réponses")
    for n in sorted(reference_counts):
        print(f"{n}: {reference_counts[n]}")

    print("\nPRÉDICTION — nombre de réponses produites")
    for n in sorted(prediction_counts):
        print(f"{n}: {prediction_counts[n]}")


print_prediction_cardinality(
    "BASE — MULTI",
    [r for r in base_results if r["cardinality"] == "MULTI"],
)

print_prediction_cardinality(
    "SFT v4 — MULTI",
    [r for r in sft_results if r["cardinality"] == "MULTI"],
)

print_breakdown(
    "BASE — SINGLE vs MULTI",
    base_results,
    "cardinality",
)

print_breakdown(
    "SFT v4 — SINGLE vs MULTI",
    sft_results,
    "cardinality",
)
def print_cardinality_matrix(name, results):

    matrix = defaultdict(Counter)

    for r in results:
        if r["cardinality"] != "MULTI":
            continue

        ref_n = len(r["reference"])
        pred_n = len(r["prediction"])

        matrix[ref_n][pred_n] += 1

    print(f"\n{name} — MATRICE RÉFÉRENCE → PRÉDICTION")
    print("=" * 70)

    prediction_sizes = sorted({
        pred_n
        for row in matrix.values()
        for pred_n in row
    })

    print(
        f"{'REF':>8}"
        + "".join(f"{n:>8}" for n in prediction_sizes)
    )

    for ref_n in sorted(matrix):
        print(
            f"{ref_n:>8}"
            + "".join(
                f"{matrix[ref_n][pred_n]:>8}"
                for pred_n in prediction_sizes
            )
        )


print_cardinality_matrix("BASE", base_results)
print_cardinality_matrix("SFT v4", sft_results)
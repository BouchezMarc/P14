import json
from collections import defaultdict

TEST_FILE = "data/processed/sft_test.jsonl"
BASE_EVAL_FILE = "results/base/predictions.jsonl"
SFT_EVAL_FILE = "results/sft_v4/predictions.jsonl"


def load_jsonl(path):
    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["id"]] = record

    return records


def normalize(text):
    return " ".join(text.strip().lower().split())


def token_f1(prediction, reference):
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = defaultdict(int)
    ref_counts = defaultdict(int)

    for token in pred_tokens:
        pred_counts[token] += 1

    for token in ref_tokens:
        ref_counts[token] += 1

    common = sum(
        min(pred_counts[token], ref_counts[token])
        for token in pred_counts
    )

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return normalize(prediction) == normalize(reference)


test = load_jsonl(TEST_FILE)
base = load_jsonl(BASE_EVAL_FILE)
sft = load_jsonl(SFT_EVAL_FILE)


# ---------------------------------------------------------------------
# Sélection QA
# ---------------------------------------------------------------------

qa_ids = [
    record_id
    for record_id, record in test.items()
    if record.get("task") == "medical_qa"
]

print("=" * 80)
print("ANALYSE QA — BASE vs SFT v4")
print("=" * 80)

print(f"Test total : {len(test)}")
print(f"QA        : {len(qa_ids)}")


# ---------------------------------------------------------------------
# Calcul des métriques
# ---------------------------------------------------------------------

results = []

for record_id in qa_ids:

    reference = test[record_id]["response"]

    base_prediction = base[record_id]["prediction"]
    sft_prediction = sft[record_id]["prediction"]

    base_f1 = token_f1(base_prediction, reference)
    sft_f1 = token_f1(sft_prediction, reference)

    base_em = exact_match(base_prediction, reference)
    sft_em = exact_match(sft_prediction, reference)

    results.append({
        "id": record_id,
        "language": test[record_id].get("language"),
        "source": test[record_id].get("source", {}).get("dataset"),
        "reference": reference,
        "base_prediction": base_prediction,
        "sft_prediction": sft_prediction,
        "base_f1": base_f1,
        "sft_f1": sft_f1,
        "delta_f1": sft_f1 - base_f1,
        "base_em": base_em,
        "sft_em": sft_em,
        "base_len": len(base_prediction.split()),
        "sft_len": len(sft_prediction.split()),
    })


# ---------------------------------------------------------------------
# Résumé global QA
# ---------------------------------------------------------------------

def summarize(items):

    n = len(items)

    base_f1 = sum(x["base_f1"] for x in items) / n
    sft_f1 = sum(x["sft_f1"] for x in items) / n

    base_em = sum(x["base_em"] for x in items) / n
    sft_em = sum(x["sft_em"] for x in items) / n

    base_len = sum(x["base_len"] for x in items) / n
    sft_len = sum(x["sft_len"] for x in items) / n

    return {
        "n": n,
        "base_f1": base_f1,
        "sft_f1": sft_f1,
        "delta_f1": sft_f1 - base_f1,
        "base_em": base_em,
        "sft_em": sft_em,
        "delta_em": sft_em - base_em,
        "base_len": base_len,
        "sft_len": sft_len,
    }


summary = summarize(results)

print("\n" + "-" * 80)
print("QA GLOBAL")
print("-" * 80)

print(f"Examples          : {summary['n']}")
print(f"Base Token F1     : {summary['base_f1']:.4f}")
print(f"SFT v4 Token F1   : {summary['sft_f1']:.4f}")
print(f"Delta F1          : {summary['delta_f1']:+.4f}")

print(f"\nBase Exact Match  : {summary['base_em']:.4f}")
print(f"SFT v4 Exact Match: {summary['sft_em']:.4f}")
print(f"Delta EM          : {summary['delta_em']:+.4f}")

print(f"\nBase longueur moy.: {summary['base_len']:.1f} tokens")
print(f"SFT v4 longueur   : {summary['sft_len']:.1f} tokens")


# ---------------------------------------------------------------------
# Par source
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PAR SOURCE")
print("=" * 80)

by_source = defaultdict(list)

for item in results:
    by_source[item["source"]].append(item)

for source, items in sorted(by_source.items()):

    s = summarize(items)

    print(f"\n{source}")
    print(f"  Examples       : {s['n']}")
    print(f"  Base F1        : {s['base_f1']:.4f}")
    print(f"  SFT v4 F1      : {s['sft_f1']:.4f}")
    print(f"  Delta F1       : {s['delta_f1']:+.4f}")
    print(f"  Base EM        : {s['base_em']:.4f}")
    print(f"  SFT v4 EM      : {s['sft_em']:.4f}")


# ---------------------------------------------------------------------
# Cas où SFT v4 améliore / régresse
# ---------------------------------------------------------------------

improved = [x for x in results if x["delta_f1"] > 0]
regressed = [x for x in results if x["delta_f1"] < 0]
unchanged = [x for x in results if x["delta_f1"] == 0]

print("\n" + "=" * 80)
print("EVOLUTION BASE → SFT v4")
print("=" * 80)

print(f"Améliorations : {len(improved)}")
print(f"Régressions   : {len(regressed)}")
print(f"Identiques    : {len(unchanged)}")


# ---------------------------------------------------------------------
# 10 plus grosses régressions
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PLUS GROSSES RÉGRESSIONS")
print("=" * 80)

for item in sorted(regressed, key=lambda x: x["delta_f1"])[:10]:

    print("\n" + "-" * 80)
    print("ID:", item["id"])
    print("Source:", item["source"])

    print(f"Base F1 : {item['base_f1']:.4f}")
    print(f"SFT  F1 : {item['sft_f1']:.4f}")
    print(f"Delta   : {item['delta_f1']:+.4f}")

    print("\nREFERENCE:")
    print(item["reference"])

    print("\nBASE:")
    print(item["base_prediction"])

    print("\nSFT v4:")
    print(item["sft_prediction"])


# ---------------------------------------------------------------------
# 10 plus grosses améliorations
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PLUS GROSSES AMÉLIORATIONS")
print("=" * 80)

for item in sorted(improved, key=lambda x: x["delta_f1"], reverse=True)[:10]:

    print("\n" + "-" * 80)
    print("ID:", item["id"])
    print("Source:", item["source"])

    print(f"Base F1 : {item['base_f1']:.4f}")
    print(f"SFT  F1 : {item['sft_f1']:.4f}")
    print(f"Delta   : {item['delta_f1']:+.4f}")

    print("\nREFERENCE:")
    print(item["reference"])

    print("\nBASE:")
    print(item["base_prediction"])

    print("\nSFT v4:")
    print(item["sft_prediction"])
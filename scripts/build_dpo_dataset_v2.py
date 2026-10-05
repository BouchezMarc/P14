import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RAW_DIR = Path("data/raw/ultramedical_preference")
DEFAULT_PROCESSED_DIR = Path("data/processed/dpo_dataset_v2")
DEFAULT_SFT_FILE = Path("data/processed/sft_5000.jsonl")

TRAIN_SIZE = 5000
VALIDATION_SIZE = 1000

SEED = 42

PROCESSING_VERSION = "dpo_v6"

DPO_LABELS = {"easy", "hard", "length"}

REWARDBENCH_LABELS = {"easy", "hard", "length", "human"}

REWARDBENCH_EXPECTED = {
    "easy": 237,
    "hard": 196,
    "length": 180,
    "human": 163,
}

KNOWN_REWARDBENCH_DUPLICATE = (
    "A 5-year old girl presents with hypeension and virilization. "
    "There is also finding of hypokalemia what is the diagnosis-\n\n"
    "A. 21-hydroxylase deficiency\n"
    "B. 3-13 hydroxy steroid deficeicny\n"
    "C. 11-13 hydroxylase deficeincy\n"
    "D. Conn's disease"
)


# ============================================================
# UTILITAIRES JSON
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {path} ligne {line_number}: {exc}"
                ) from exc

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


# ============================================================
# NORMALISATION
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_conversation(conversation):
    """
    Retourne une liste normalisée de messages :
    [{"role": "...", "content": "..."}]
    """

    if not isinstance(conversation, list):
        return []

    normalized = []

    for message in conversation:
        if not isinstance(message, dict):
            continue

        role = normalize_text(message.get("role"))
        content = normalize_text(message.get("content"))

        if not role or not content:
            continue

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized


def conversation_text(conversation):
    messages = normalize_conversation(conversation)

    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages
    )


def get_prompt_from_conversation(conversation):
    messages = normalize_conversation(conversation)

    for message in messages:
        if message["role"] == "user":
            return message["content"]

    return ""


def get_assistant_text(conversation):
    messages = normalize_conversation(conversation)

    for message in messages:
        if message["role"] == "assistant":
            return message["content"]

    return ""


# ============================================================
# VALIDATION STRUCTURELLE
# ============================================================

def valid_conversation(conversation):
    messages = normalize_conversation(conversation)

    if len(messages) < 2:
        return False

    has_user = any(
        m["role"] == "user"
        for m in messages
    )

    has_assistant = any(
        m["role"] == "assistant"
        for m in messages
    )

    return has_user and has_assistant


def valid_preference(row):
    if not isinstance(row, dict):
        return False

    if not isinstance(row.get("prompt"), str):
        return False

    if not row["prompt"].strip():
        return False

    if not valid_conversation(row.get("chosen")):
        return False

    if not valid_conversation(row.get("rejected")):
        return False

    chosen_text = get_assistant_text(row["chosen"])
    rejected_text = get_assistant_text(row["rejected"])

    if not chosen_text or not rejected_text:
        return False

    if chosen_text == rejected_text:
        return False

    return True


# ============================================================
# IDENTIFIANTS STABLES
# ============================================================

def stable_hash(*values):
    payload = "\n".join(
        normalize_text(str(value))
        for value in values
    )

    return hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()


def preference_id(row):
    source_id = row.get("prompt_id", "")

    return stable_hash(
        source_id,
        row.get("prompt", ""),
        conversation_text(row.get("chosen", [])),
        conversation_text(row.get("rejected", [])),
    )


# ============================================================
# EXTRACTION DES INFORMATIONS SOURCE
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        if value is None:
            return None

        return int(value)

    except (TypeError, ValueError):
        return None


def extract_source_metadata(row):
    """
    Les informations suivantes servent UNIQUEMENT à la sélection.

    Elles ne seront PAS écrites dans le dataset final.
    """

    metadata = row.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    chosen_meta = metadata.get("chosen")
    rejected_meta = metadata.get("rejected")

    if not isinstance(chosen_meta, dict):
        chosen_meta = {}

    if not isinstance(rejected_meta, dict):
        rejected_meta = {}

    chosen_score = safe_float(
        chosen_meta.get("score")
    )

    rejected_score = safe_float(
        rejected_meta.get("score")
    )

    chosen_rank = safe_int(
        chosen_meta.get("rank")
    )

    rejected_rank = safe_int(
        rejected_meta.get("rank")
    )

    chosen_evaluation = normalize_text(
        chosen_meta.get("evaluation", "")
    )

    rejected_evaluation = normalize_text(
        rejected_meta.get("evaluation", "")
    )

    feedback = normalize_text(
        row.get("feedback", "")
    )

    golden_answer = metadata.get("golden_answer")

    if golden_answer is not None:
        golden_answer = normalize_text(
            golden_answer
        )

    return {
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "chosen_rank": chosen_rank,
        "rejected_rank": rejected_rank,
        "chosen_evaluation": chosen_evaluation,
        "rejected_evaluation": rejected_evaluation,
        "feedback": feedback,
        "golden_answer": golden_answer,
    }


# ============================================================
# SCORE DE SÉLECTION
# ============================================================

POSITIVE_TERMS = {
    "accurate": 2.0,
    "accuracy": 2.0,
    "correct": 2.0,
    "correctly": 2.0,
    "evidence-based": 2.0,
    "evidence based": 2.0,
    "relevant": 1.5,
    "clinically relevant": 2.0,
    "appropriate": 1.0,
    "appropriate terminology": 1.5,
    "comprehensive": 1.0,
    "thorough": 1.0,
    "detailed": 0.75,
    "well-structured": 0.75,
    "well structured": 0.75,
    "informative": 0.75,
    "addresses": 0.5,
    "adheres": 1.0,
    "clear": 0.5,
}

NEGATIVE_TERMS = {
    "incorrect": -2.0,
    "inaccurate": -2.0,
    "error": -1.5,
    "errors": -1.5,
    "misleading": -2.0,
    "irrelevant": -2.0,
    "not relevant": -2.0,
    "hallucination": -2.5,
    "hallucinations": -2.5,
    "unsupported": -2.0,
    "unsafe": -2.5,
    "dangerous": -2.5,
    "wrong": -2.0,
    "fails to address": -1.5,
    "fails to answer": -1.5,
}


def feedback_quality_score(text):
    """
    Analyse lexicale du feedback existant.

    Ce n'est PAS une évaluation médicale.
    """

    text_lower = normalize_text(text).lower()

    score = 0.0

    for term, weight in POSITIVE_TERMS.items():
        if term in text_lower:
            score += weight

    for term, weight in NEGATIVE_TERMS.items():
        if term in text_lower:
            score += weight

    return score


def preference_selection_score(row):
    """
    Score interne utilisé uniquement pour la sélection.

    Plus il est élevé, plus la préférence source est forte
    et documentée.

    Ce score n'est jamais écrit dans le JSON final.
    """

    source = extract_source_metadata(row)

    chosen_score = source["chosen_score"]
    rejected_score = source["rejected_score"]

    chosen_rank = source["chosen_rank"]
    rejected_rank = source["rejected_rank"]

    score = 0.0

    # --------------------------------------------------------
    # 1. Différence de score GPT
    # --------------------------------------------------------

    if (
        chosen_score is not None
        and rejected_score is not None
    ):
        margin = chosen_score - rejected_score

        score += margin * 5.0
        score += chosen_score * 1.0

        if chosen_score < 4:
            score -= 3.0

    # --------------------------------------------------------
    # 2. Cohérence des rangs
    # --------------------------------------------------------

    if (
        chosen_rank is not None
        and rejected_rank is not None
    ):
        rank_gap = rejected_rank - chosen_rank

        if rank_gap > 0:
            score += min(rank_gap, 5) * 1.5

        elif rank_gap < 0:
            score -= 5.0

    # --------------------------------------------------------
    # 3. Evaluation de chosen
    # --------------------------------------------------------

    score += (
        feedback_quality_score(
            source["chosen_evaluation"]
        )
        * 1.5
    )

    # --------------------------------------------------------
    # 4. Evaluation de rejected
    # --------------------------------------------------------

    rejected_eval = source["rejected_evaluation"]

    negative_rejected = 0.0

    rejected_lower = rejected_eval.lower()

    for term, weight in NEGATIVE_TERMS.items():
        if term in rejected_lower:
            negative_rejected += abs(weight)

    score += min(
        negative_rejected,
        5.0,
    )

    # --------------------------------------------------------
    # 5. Feedback global
    # --------------------------------------------------------

    score += (
        feedback_quality_score(
            source["feedback"]
        )
        * 1.0
    )

    # --------------------------------------------------------
    # 6. Verdict explicite
    # --------------------------------------------------------

    feedback_lower = source["feedback"].lower()

    if "final verdict: [[a]]" in feedback_lower:
        score += 3.0

    elif "final verdict: [[b]]" in feedback_lower:
        score -= 8.0

    return score


# ============================================================
# VALIDITÉ DE LA PRÉFÉRENCE SOURCE
# ============================================================

def source_preference_is_coherent(row):
    """
    Vérifie que les informations disponibles dans la source
    ne contredisent pas la préférence chosen > rejected.
    """

    source = extract_source_metadata(row)

    chosen_score = source["chosen_score"]
    rejected_score = source["rejected_score"]

    chosen_rank = source["chosen_rank"]
    rejected_rank = source["rejected_rank"]

    if (
        chosen_score is not None
        and rejected_score is not None
        and chosen_score < rejected_score
    ):
        return False

    if (
        chosen_rank is not None
        and rejected_rank is not None
        and chosen_rank > rejected_rank
    ):
        return False

    feedback = source["feedback"].lower()

    if "final verdict: [[b]]" in feedback:
        return False

    return True


# ============================================================
# REWARDBENCH
# ============================================================

def extract_rewardbench_prompts(rewardbench_rows):
    prompts = set()

    for row in rewardbench_rows:

        prompt = normalize_text(
            row.get("prompt", "")
        )

        if prompt:
            prompts.add(prompt)

    return prompts


def build_rewardbench(
    raw_dir: Path,
    processed_dir: Path,
):
    """
    Construit RewardBench à partir du test source,
    comme dans le pipeline précédent.

    Le fichier exact peut varier selon la version du dépôt.
    """

    candidates = [
        raw_dir / "test.json",
        raw_dir / "test.jsonl",
        processed_dir / "medical_rewardbench.jsonl",
    ]

    source_file = None

    for candidate in candidates:
        if candidate.exists():
            source_file = candidate
            break

    if source_file is None:
        raise FileNotFoundError(
            "Impossible de trouver le fichier RewardBench/test."
        )

    if source_file.suffix == ".jsonl":
        rows = load_jsonl(source_file)

    else:
        rows = load_json(source_file)

    if not isinstance(rows, list):
        raise ValueError(
            f"Format RewardBench inattendu : {source_file}"
        )

    rewardbench_rows = []

    seen = set()
    duplicate_rows = []

    for row in rows:

        if not isinstance(row, dict):
            continue

        prompt = normalize_text(
            row.get("prompt", "")
        )

        if not prompt:
            continue

        key = stable_hash(prompt)

        if key in seen:
            duplicate_rows.append(row)
            continue

        seen.add(key)
        rewardbench_rows.append(row)

    # --------------------------------------------------------
    # Vérification explicite du doublon connu
    # --------------------------------------------------------

    if duplicate_rows:

        known_duplicate = normalize_text(
            KNOWN_REWARDBENCH_DUPLICATE
        )

        known_found = any(
            normalize_text(
                row.get("prompt", "")
            ) == known_duplicate
            for row in duplicate_rows
        )

        if not known_found:
            raise RuntimeError(
                "Un doublon inattendu a été détecté "
                "dans RewardBench."
            )

        print(
            "Doublon connu RewardBench détecté et supprimé."
        )

    return rewardbench_rows


# ============================================================
# SFT PROMPTS
# ============================================================

def load_sft_prompts(path: Path):
    rows = load_jsonl(path)

    prompts = set()

    for row in rows:

        prompt = normalize_text(
            row.get("instruction", "")
        )

        if prompt:
            prompts.add(prompt)

    return prompts


# ============================================================
# NETTOYAGE DES CANDIDATS
# ============================================================

def prepare_candidates(
    raw_rows,
    sft_prompts,
    rewardbench_prompts,
):
    candidates = []

    counters = Counter()

    seen_ids = set()
    seen_pairs = set()

    for row in raw_rows:

        counters["raw"] += 1

        if not valid_preference(row):
            counters["invalid"] += 1
            continue

        label_type = normalize_text(
            row.get("label_type", "")
        ).lower()

        if label_type not in DPO_LABELS:
            counters["invalid_label"] += 1
            continue

        prompt = normalize_text(
            row.get("prompt", "")
        )

        prompt_id = normalize_text(
            row.get("prompt_id", "")
        )

        # ----------------------------------------------------
        # Exclusion SFT
        # ----------------------------------------------------

        if prompt in sft_prompts:
            counters["sft_leakage"] += 1
            continue

        # ----------------------------------------------------
        # Exclusion RewardBench
        # ----------------------------------------------------

        if prompt in rewardbench_prompts:
            counters["rewardbench_leakage"] += 1
            continue

        # ----------------------------------------------------
        # Cohérence préférence
        # ----------------------------------------------------

        if not source_preference_is_coherent(row):
            counters["incoherent_preference"] += 1
            continue

        # ----------------------------------------------------
        # ID stable
        # ----------------------------------------------------

        source_id = (
            prompt_id
            if prompt_id
            else preference_id(row)
        )

        stable_id = stable_hash(
            source_id,
            prompt,
            conversation_text(row["chosen"]),
            conversation_text(row["rejected"]),
        )

        if stable_id in seen_ids:
            counters["duplicate_id"] += 1
            continue

        seen_ids.add(stable_id)

        # ----------------------------------------------------
        # Doublon exact de paire
        # ----------------------------------------------------

        chosen_text = get_assistant_text(
            row["chosen"]
        )

        rejected_text = get_assistant_text(
            row["rejected"]
        )

        pair_key = stable_hash(
            prompt,
            chosen_text,
            rejected_text,
        )

        if pair_key in seen_pairs:
            counters["duplicate_pair"] += 1
            continue

        seen_pairs.add(pair_key)

        # ----------------------------------------------------
        # Préparation interne
        # ----------------------------------------------------

        candidate = {
            "raw": row,
            "prompt": prompt,
            "label_type": label_type,
            "source_id": source_id,
            "stable_id": stable_id,
            "selection_score": preference_selection_score(row),
        }

        candidates.append(candidate)

        counters["usable"] += 1

    return candidates, counters


# ============================================================
# ALLOCATION STRATIFIÉE
# ============================================================

def proportional_quotas(
    counts,
    total,
):
    """
    Répartition proportionnelle avec méthode des plus grands restes.

    Garantit :
        sum(quotas) == total
    """

    labels = sorted(counts)

    total_available = sum(
        counts.values()
    )

    if total_available == 0:
        raise ValueError(
            "Aucun candidat disponible."
        )

    if total > total_available:
        raise ValueError(
            f"Impossible de sélectionner {total} exemples "
            f"parmi {total_available}."
        )

    exact = {
        label: (
            counts[label]
            / total_available
            * total
        )
        for label in labels
    }

    quotas = {
        label: math.floor(exact[label])
        for label in labels
    }

    remaining = (
        total
        - sum(quotas.values())
    )

    remainders = sorted(
        labels,
        key=lambda label: (
            exact[label] - quotas[label],
            counts[label],
            label,
        ),
        reverse=True,
    )

    for label in remainders[:remaining]:
        quotas[label] += 1

    return quotas


def stratified_select(
    candidates,
    total,
    seed,
):
    """
    Sélection stratifiée par label_type.

    Dans chaque strate :
    - mélange reproductible ;
    - classement par score de sélection ;
    - sélection du quota.

    Le score n'est PAS écrit dans le dataset final.
    """

    by_label = defaultdict(list)

    for candidate in candidates:
        by_label[
            candidate["label_type"]
        ].append(candidate)

    counts = {
        label: len(rows)
        for label, rows in by_label.items()
    }

    quotas = proportional_quotas(
        counts,
        total,
    )

    rng = np.random.default_rng(seed)

    selected = []

    for label in sorted(by_label):

        rows = list(
            by_label[label]
        )

        # Mélange reproductible.
        rng.shuffle(rows)

        # Classement par force documentaire
        # de la préférence source.
        rows.sort(
            key=lambda x: x["selection_score"],
            reverse=True,
        )

        quota = quotas.get(
            label,
            0,
        )

        selected.extend(
            rows[:quota]
        )

    # Mélange final reproductible.
    rng.shuffle(selected)

    return selected, quotas


# ============================================================
# SPLIT TRAIN / VALIDATION
# ============================================================

def split_candidates(
    candidates,
    train_size,
    validation_size,
    seed,
):
    """
    Split stratifié reproductible.

    IMPORTANT :
    l'exclusion train/validation est faite sur le PROMPT,
    pas seulement sur stable_id.

    Plusieurs paires UltraMedical peuvent avoir le même
    prompt avec des chosen/rejected différents. Elles doivent
    toutes rester dans le même split.
    """

    total = (
        train_size
        + validation_size
    )

    if len(candidates) < total:
        raise ValueError(
            f"Seulement {len(candidates)} candidats disponibles "
            f"pour {total} exemples demandés."
        )

    # --------------------------------------------------------
    # 1. Sélection validation
    # --------------------------------------------------------

    validation, validation_quotas = (
        stratified_select(
            candidates=candidates,
            total=validation_size,
            seed=seed,
        )
    )

    # --------------------------------------------------------
    # 2. PROMPTS de validation
    #
    # On exclut TOUS les candidats portant ces prompts.
    # --------------------------------------------------------

    validation_prompts = {
        candidate["prompt"]
        for candidate in validation
    }

    # --------------------------------------------------------
    # 3. Candidats restant pour train
    #
    # L'ancien code utilisait seulement stable_id.
    # C'était insuffisant : deux IDs différents peuvent
    # correspondre au même prompt.
    # --------------------------------------------------------

    remaining = [
        candidate
        for candidate in candidates
        if candidate["prompt"]
        not in validation_prompts
    ]

    if len(remaining) < train_size:
        raise RuntimeError(
            "Pas assez de candidats après exclusion "
            "des prompts de validation : "
            f"{len(remaining)} disponibles pour "
            f"{train_size} requis."
        )

    # --------------------------------------------------------
    # 4. Sélection train
    # --------------------------------------------------------

    train, train_quotas = (
        stratified_select(
            candidates=remaining,
            total=train_size,
            seed=seed + 1,
        )
    )

    # --------------------------------------------------------
    # 5. Contrôle explicite des prompts
    # --------------------------------------------------------

    train_prompts = {
        candidate["prompt"]
        for candidate in train
    }

    overlap = (
        validation_prompts
        & train_prompts
    )

    if overlap:
        raise RuntimeError(
            "Même prompt présent dans train et validation : "
            f"{len(overlap)} prompts."
        )

    return (
        train,
        validation,
        train_quotas,
        validation_quotas,
    )


# ============================================================
# CONVERSION VERS LE FORMAT FINAL
# ============================================================

def make_final_row(
    candidate,
    split,
):
    row = candidate["raw"]

    source_id = normalize_text(
        row.get("prompt_id", "")
    )

    return {
        "id": (
            f"dpo_{split}_"
            f"{candidate['stable_id'][:16]}"
        ),

        "prompt": normalize_text(
            row["prompt"]
        ),

        "chosen": normalize_conversation(
            row["chosen"]
        ),

        "rejected": normalize_conversation(
            row["rejected"]
        ),

        "source": {
            "dataset": "UltraMedical-Preference",
            "split": normalize_text(
                row.get("split", "train")
            ),
            "label_type": normalize_text(
                row["label_type"]
            ).lower(),
        },

        "quality": {
            "preference_valid": True,
            "clinically_validated": False,
            "clinical_validation_status": "pending",
        },

        "provenance": {
            "source_id": source_id,
            "processing_version": PROCESSING_VERSION,
            "dpo_split": split,
        },
    }


# ============================================================
# CONTROLES FINAUX
# ============================================================

def check_final_dataset(
    train_rows,
    validation_rows,
    sft_prompts,
    rewardbench_prompts,
):
    train_ids = {
        row["id"]
        for row in train_rows
    }

    validation_ids = {
        row["id"]
        for row in validation_rows
    }

    if train_ids & validation_ids:
        raise RuntimeError(
            "Collision d'ID entre train et validation."
        )

    train_prompts = {
        normalize_text(row["prompt"])
        for row in train_rows
    }

    validation_prompts = {
        normalize_text(row["prompt"])
        for row in validation_rows
    }

    if train_prompts & validation_prompts:
        raise RuntimeError(
            "Même prompt présent dans train et validation."
        )

    if train_prompts & sft_prompts:
        raise RuntimeError(
            "Fuite SFT détectée dans train."
        )

    if validation_prompts & sft_prompts:
        raise RuntimeError(
            "Fuite SFT détectée dans validation."
        )

    if train_prompts & rewardbench_prompts:
        raise RuntimeError(
            "Fuite RewardBench détectée dans train."
        )

    if validation_prompts & rewardbench_prompts:
        raise RuntimeError(
            "Fuite RewardBench détectée dans validation."
        )

    expected_keys = {
        "id",
        "prompt",
        "chosen",
        "rejected",
        "source",
        "quality",
        "provenance",
    }

    for split_name, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
    ):
        for row in rows:

            if set(row.keys()) != expected_keys:
                raise RuntimeError(
                    f"Schéma incorrect dans {split_name}: "
                    f"{set(row.keys())}"
                )

            if set(row["source"].keys()) != {
                "dataset",
                "split",
                "label_type",
            }:
                raise RuntimeError(
                    "Modification inattendue de source."
                )

            if set(row["quality"].keys()) != {
                "preference_valid",
                "clinically_validated",
                "clinical_validation_status",
            }:
                raise RuntimeError(
                    "Modification inattendue de quality."
                )

            if set(row["provenance"].keys()) != {
                "source_id",
                "processing_version",
                "dpo_split",
            }:
                raise RuntimeError(
                    "Modification inattendue de provenance."
                )


# ============================================================
# RAPPORT
# ============================================================

def print_distribution(
    name,
    rows,
):
    counts = Counter(
        row["source"]["label_type"]
        for row in rows
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Total : {len(rows)}"
    )

    for label in sorted(counts):
        count = counts[label]

        pct = (
            count
            / len(rows)
            * 100
        )

        print(
            f"  {label:8s} : {count:5d} "
            f"({pct:6.2f} %)"
        )


def print_selection_quality(
    candidates,
    name,
):
    scores = [
        candidate["selection_score"]
        for candidate in candidates
    ]

    if not scores:
        return

    print()
    print(
        f"{name} - sélection interne"
    )

    print(
        f"  score min : {min(scores):.3f}"
    )

    print(
        f"  score max : {max(scores):.3f}"
    )

    print(
        f"  score moy : "
        f"{sum(scores) / len(scores):.3f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Construction reproductible du dataset DPO "
            "à partir d'UltraMedical-Preference."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    parser.add_argument(
        "--sft-file",
        type=Path,
        default=DEFAULT_SFT_FILE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=TRAIN_SIZE,
    )

    parser.add_argument(
        "--validation-size",
        type=int,
        default=VALIDATION_SIZE,
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir
    processed_dir = args.processed_dir

    # --------------------------------------------------------
    # Fichiers source
    # --------------------------------------------------------

    train_source = raw_dir / "train.json"
    dev_source = raw_dir / "dev.json"

    if not train_source.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {train_source}"
        )

    if not dev_source.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {dev_source}"
        )

    print("=" * 70)
    print("DPO DATASET BUILDER")
    print("=" * 70)

    print(
        f"Version        : {PROCESSING_VERSION}"
    )

    print(
        f"Seed           : {args.seed}"
    )

    print(
        f"Train          : {args.train_size}"
    )

    print(
        f"Validation     : {args.validation_size}"
    )

    # --------------------------------------------------------
    # Chargement SFT
    # --------------------------------------------------------

    print()
    print("Chargement SFT...")

    sft_prompts = load_sft_prompts(
        args.sft_file
    )

    print(
        f"Prompts SFT : {len(sft_prompts)}"
    )

    # --------------------------------------------------------
    # RewardBench
    # --------------------------------------------------------

    print()
    print("Chargement RewardBench...")

    rewardbench_rows = build_rewardbench(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    rewardbench_prompts = (
        extract_rewardbench_prompts(
            rewardbench_rows
        )
    )

    print(
        f"RewardBench : "
        f"{len(rewardbench_prompts)}"
    )

    # --------------------------------------------------------
    # Chargement UltraMedical
    # --------------------------------------------------------

    print()
    print(
        "Chargement UltraMedical-Preference..."
    )

    train_raw = load_json(
        train_source
    )

    dev_raw = load_json(
        dev_source
    )

    if not isinstance(train_raw, list):
        raise ValueError(
            "train.json doit contenir une liste."
        )

    if not isinstance(dev_raw, list):
        raise ValueError(
            "dev.json doit contenir une liste."
        )

    # --------------------------------------------------------
    # Combinaison train + dev
    # --------------------------------------------------------

    raw_rows = []

    for row in train_raw:

        if isinstance(row, dict):
            row = dict(row)
            row["split"] = "train"
            raw_rows.append(row)

    for row in dev_raw:

        if isinstance(row, dict):
            row = dict(row)
            row["split"] = "dev"
            raw_rows.append(row)

    print(
        f"Entrées source : "
        f"{len(raw_rows)}"
    )

    # --------------------------------------------------------
    # Nettoyage + sélection interne
    # --------------------------------------------------------

    candidates, counters = (
        prepare_candidates(
            raw_rows=raw_rows,
            sft_prompts=sft_prompts,
            rewardbench_prompts=rewardbench_prompts,
        )
    )

    print()
    print("=" * 70)
    print("NETTOYAGE")
    print("=" * 70)

    for key in sorted(counters):
        print(
            f"{key:28s}: {counters[key]}"
        )

    # --------------------------------------------------------
    # Contrôle volume
    # --------------------------------------------------------

    required = (
        args.train_size
        + args.validation_size
    )

    if len(candidates) < required:
        raise RuntimeError(
            f"Après filtrage, seulement "
            f"{len(candidates)} candidats "
            f"disponibles pour {required} requis."
        )

    # --------------------------------------------------------
    # Split stratifié reproductible
    # --------------------------------------------------------

    (
        train_candidates,
        validation_candidates,
        train_quotas,
        validation_quotas,
    ) = split_candidates(
        candidates=candidates,
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    print()
    print("=" * 70)
    print("QUOTAS TRAIN")
    print("=" * 70)

    for label in sorted(train_quotas):
        print(
            f"{label:8s}: "
            f"{train_quotas[label]}"
        )

    print()
    print("=" * 70)
    print("QUOTAS VALIDATION")
    print("=" * 70)

    for label in sorted(
        validation_quotas
    ):
        print(
            f"{label:8s}: "
            f"{validation_quotas[label]}"
        )

    # --------------------------------------------------------
    # Conversion finale
    # --------------------------------------------------------

    train_rows = [
        make_final_row(
            candidate,
            "train",
        )
        for candidate in train_candidates
    ]

    validation_rows = [
        make_final_row(
            candidate,
            "validation",
        )
        for candidate in validation_candidates
    ]

    # --------------------------------------------------------
    # Contrôles finaux
    # --------------------------------------------------------

    check_final_dataset(
        train_rows=train_rows,
        validation_rows=validation_rows,
        sft_prompts=sft_prompts,
        rewardbench_prompts=rewardbench_prompts,
    )

    # --------------------------------------------------------
    # Contrôle tailles exactes
    # --------------------------------------------------------

    if len(train_rows) != args.train_size:
        raise RuntimeError(
            f"Train incorrect : "
            f"{len(train_rows)} au lieu de "
            f"{args.train_size}."
        )

    if len(validation_rows) != args.validation_size:
        raise RuntimeError(
            f"Validation incorrecte : "
            f"{len(validation_rows)} au lieu de "
            f"{args.validation_size}."
        )

    # --------------------------------------------------------
    # Sorties
    # --------------------------------------------------------

    train_output = (
        processed_dir        
        / "dpo_train.jsonl"
    )

    validation_output = (
        processed_dir        
        / "dpo_validation.jsonl"
    )

    write_jsonl(
        train_output,
        train_rows,
    )

    write_jsonl(
        validation_output,
        validation_rows,
    )

    # --------------------------------------------------------
    # Rapports
    # --------------------------------------------------------

    print_distribution(
        "DPO TRAIN",
        train_rows,
    )

    print_distribution(
        "DPO VALIDATION",
        validation_rows,
    )

    print_selection_quality(
        train_candidates,
        "TRAIN",
    )

    print_selection_quality(
        validation_candidates,
        "VALIDATION",
    )

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)

    print(
        f"Train      : {train_output}"
    )

    print(
        f"Validation : {validation_output}"
    )

    print()
    print(
        "Format final conservé : "
        "id / prompt / chosen / rejected / source / "
        "quality / provenance"
    )

    print(
        "Les scores, ranks, evaluations et feedback "
        "source ont uniquement servi à la sélection."
    )


if __name__ == "__main__":
    main()
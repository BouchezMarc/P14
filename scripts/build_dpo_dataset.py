from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DEFAULT_RAW_DIR = Path("data/raw/ultramedical_preference")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_SFT = DEFAULT_PROCESSED_DIR / "sft_5000.jsonl"

PROCESSING_VERSION = "dpo_v5"
REWARDBENCH_VERSION = "rewardbench_v5"

DPO_LABELS = {
    "easy",
    "hard",
    "length",
}

REWARDBENCH_LABELS = {
    "easy",
    "hard",
    "length",
    "human",
}


# Doublon connu dans test.json
KNOWN_REWARDBENCH_DUPLICATE = (
    "A 5-year old girl presents with hypeension and virilization. "
    "There is also finding of hypokalemia what is the diagnosis-\n\n"
    "A. 21-hydroxylase deficiency\n"
    "B. 3-13 hydroxy steroid deficeicny\n"
    "C. 11-13 hydroxylase deficeincy\n"
    "D. Conn's disease"
)


# ============================================================
# Fichiers
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {path}, ligne {line_no}: {exc}"
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
                )
                + "\n"
            )


# ============================================================
# Normalisation
# ============================================================

def normalize_text(value) -> str:
    """
    Normalisation utilisée pour les comparaisons de prompts.

    Cette fonction doit rester cohérente avec control_dpo.py.
    """
    if not isinstance(value, str):
        return ""

    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)

    return value


def normalize_conversation(value) -> str:
    """
    Normalise une conversation :

    [
        {"role": "...", "content": "..."},
        ...
    ]
    """
    if not isinstance(value, list):
        return ""

    normalized_messages = []

    for message in value:
        if not isinstance(message, dict):
            continue

        role = normalize_text(message.get("role"))
        content = normalize_text(message.get("content"))

        if role or content:
            normalized_messages.append(
                f"{role}:{content}"
            )

    return "\n".join(normalized_messages)


def normalize_preference_value(value) -> str:
    """
    Accepte :
        - chaîne
        - conversation sous forme de liste
    """
    if isinstance(value, str):
        return normalize_text(value)

    if isinstance(value, list):
        return normalize_conversation(value)

    return ""


def prompt_key(value):
    return normalize_text(value)


def preference_key(row):
    """
    Identifie une préférence exacte :
        prompt + chosen + rejected
    """

    return (
        normalize_preference_value(
            row.get("prompt")
        ),
        normalize_preference_value(
            row.get("chosen")
        ),
        normalize_preference_value(
            row.get("rejected")
        ),
    )


# ============================================================
# Validation du contenu source
# ============================================================

def valid_conversation(value) -> bool:
    """
    Valide une conversation :

    [
        {"role": "...", "content": "..."},
        ...
    ]
    """

    if not isinstance(value, list):
        return False

    if not value:
        return False

    for message in value:
        if not isinstance(message, dict):
            return False

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str):
            return False

        if not role.strip():
            return False

        if not isinstance(content, str):
            return False

        if not content.strip():
            return False

    return True


def valid_preference_value(value) -> bool:
    """
    Accepte :
        - chaîne non vide
        - conversation non vide
    """

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return valid_conversation(value)

    return False


def valid_preference(row) -> bool:
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")

    if not isinstance(prompt, str):
        return False

    if not prompt.strip():
        return False

    if not valid_preference_value(chosen):
        return False

    if not valid_preference_value(rejected):
        return False

    if (
        normalize_preference_value(chosen)
        == normalize_preference_value(rejected)
    ):
        return False

    return True


# ============================================================
# IDs
# ============================================================

def source_id(row):
    """
    UltraMedical-Preference utilise principalement prompt_id.

    Fallback sur id si présent.
    """
    value = row.get("prompt_id")

    if value is None:
        value = row.get("id")

    if value is None:
        return None

    return str(value)


def stable_id(prefix, row):
    """
    ID stable basé uniquement sur le contenu.

    L'index n'est volontairement PAS utilisé :
    deux générations du dataset avec le même contenu
    produisent les mêmes IDs.
    """

    raw = json.dumps(
        {
            "prompt": row.get("prompt"),
            "chosen": row.get("chosen"),
            "rejected": row.get("rejected"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return f"{prefix}_{digest}"


# ============================================================
# RewardBench
# ============================================================

def build_rewardbench(test_rows):
    """
    Construit medical_rewardbench.jsonl à partir de test.json.

    Source attendue :
        777 lignes

    Après suppression du doublon exact connu :
        776 lignes uniques

    Distribution attendue :
        easy   : 237
        hard   : 196
        length : 180
        human  : 163
    """

    print(
        f"RewardBench source : {len(test_rows)} lignes"
    )

    candidates = []

    for row in test_rows:

        if not valid_preference(row):
            continue

        label_type = row.get("label_type")

        if label_type not in REWARDBENCH_LABELS:
            continue

        candidates.append(row)

    # --------------------------------------------------------
    # Déduplication exacte
    # --------------------------------------------------------

    unique_rows = []
    seen_preferences = set()
    duplicate_rows = []

    for row in candidates:

        key = preference_key(row)

        if key in seen_preferences:
            duplicate_rows.append(row)
            continue

        seen_preferences.add(key)
        unique_rows.append(row)

    print(
        f"RewardBench uniques : {len(unique_rows)} lignes"
    )

    print(
        f"Doublons exacts retirés : {len(duplicate_rows)}"
    )

    # --------------------------------------------------------
    # Vérification du doublon connu
    # --------------------------------------------------------

    if duplicate_rows:

        known_found = any(
            prompt_key(row.get("prompt"))
            == prompt_key(KNOWN_REWARDBENCH_DUPLICATE)
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

    # --------------------------------------------------------
    # Vérification finale des doublons
    # --------------------------------------------------------

    final_seen = set()

    for row in unique_rows:

        key = preference_key(row)

        if key in final_seen:
            raise RuntimeError(
                "Un doublon exact subsiste dans RewardBench."
            )

        final_seen.add(key)

    # --------------------------------------------------------
    # Taille
    # --------------------------------------------------------

    expected_size = (
        len(candidates)
        - len(duplicate_rows)
    )

    if len(unique_rows) != expected_size:
        raise RuntimeError(
            "Taille RewardBench incohérente : "
            f"{len(unique_rows)} obtenues, "
            f"{expected_size} attendues."
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    distribution = Counter(
        row.get("label_type")
        for row in unique_rows
    )

    print(
        "Distribution RewardBench après déduplication :"
    )

    for label in (
        "easy",
        "hard",
        "length",
        "human",
    ):
        print(
            f"  {label:<8}: "
            f"{distribution.get(label, 0)}"
        )

    # --------------------------------------------------------
    # Vérification distribution attendue
    # --------------------------------------------------------

    expected_distribution = {
        "easy": 237,
        "hard": 196,
        "length": 180,
        "human": 163,
    }

    if distribution != expected_distribution:
        raise RuntimeError(
            "Distribution RewardBench inattendue : "
            f"{dict(distribution)} ; "
            f"attendu : {expected_distribution}"
        )

    # --------------------------------------------------------
    # Métadonnées
    # --------------------------------------------------------

    final_rows = []

    for row in unique_rows:

        final_rows.append(
            {
                "id": stable_id(
                    "rewardbench",
                    row,
                ),
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "label_type": row["label_type"],
                "source": {
                    "dataset": "UltraMedical-Preference",
                    "split": "test",
                    "label_type": row["label_type"],
                },
                "quality": {
                    "preference_valid": True,
                    "expert_reviewed_labels": True,
                    "clinical_validation_status": (
                        "expert_reviewed_labels"
                    ),
                },
                "provenance": {
                    "source_id": source_id(row),
                    "processing_version": (
                        REWARDBENCH_VERSION
                    ),
                },
            }
        )

    return final_rows


# ============================================================
# SFT
# ============================================================

def load_sft_prompts(path):
    rows = load_jsonl(path)

    return {
        prompt_key(row.get("instruction"))
        for row in rows
        if row.get("instruction")
    }


# ============================================================
# DPO
# ============================================================

def load_preference_rows(path):
    rows = load_json(path)

    if not isinstance(rows, list):
        raise ValueError(
            f"{path} doit contenir une liste JSON."
        )

    return rows


def clean_candidates(
    rows,
    excluded_prompts,
):
    """
    Prépare les candidats DPO.

    DPO :
        easy
        hard
        length

    human est réservé à RewardBench.
    """

    cleaned = []
    seen_pairs = set()

    rejected_no_label = 0
    rejected_invalid_label = 0
    rejected_invalid_content = 0
    rejected_excluded = 0
    rejected_duplicate = 0

    for row in rows:

        if not valid_preference(row):
            rejected_invalid_content += 1
            continue

        label_type = row.get("label_type")

        # ----------------------------------------------------
        # Label absent
        # ----------------------------------------------------

        if label_type is None:
            rejected_no_label += 1
            continue

        # ----------------------------------------------------
        # Label invalide
        # ----------------------------------------------------

        if label_type not in DPO_LABELS:
            rejected_invalid_label += 1
            continue

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = prompt_key(
            row.get("prompt")
        )

        if not prompt:
            rejected_invalid_content += 1
            continue

        # ----------------------------------------------------
        # SFT / RewardBench leakage
        # ----------------------------------------------------

        if prompt in excluded_prompts:
            rejected_excluded += 1
            continue

        # ----------------------------------------------------
        # Doublon exact
        # ----------------------------------------------------

        key = preference_key(row)

        if key in seen_pairs:
            rejected_duplicate += 1
            continue

        seen_pairs.add(key)

        cleaned.append(row)

    return (
        cleaned,
        rejected_no_label,
        rejected_invalid_label,
        rejected_invalid_content,
        rejected_excluded,
        rejected_duplicate,
    )


# ============================================================
# Sampling
# ============================================================

def stratified_sample(
    rows,
    size,
    seed=42,
):
    """
    Échantillonnage stratifié selon label_type.
    """

    if size <= 0:
        return []

    if len(rows) < size:
        raise RuntimeError(
            f"Pas assez de candidats : "
            f"{len(rows)} disponibles, "
            f"{size} demandés."
        )

    rng = random.Random(seed)

    groups = {}

    for row in rows:

        label = row.get("label_type")

        if label not in DPO_LABELS:
            raise RuntimeError(
                "label_type invalide dans les "
                f"candidats DPO : {label!r}"
            )

        groups.setdefault(
            label,
            [],
        ).append(row)

    labels = sorted(groups)

    # --------------------------------------------------------
    # Allocation proportionnelle
    # --------------------------------------------------------

    raw_targets = {
        label: (
            size
            * len(groups[label])
            / len(rows)
        )
        for label in labels
    }

    targets = {
        label: int(
            raw_targets[label]
        )
        for label in labels
    }

    remaining = (
        size
        - sum(targets.values())
    )

    fractions = sorted(
        labels,
        key=lambda label: (
            raw_targets[label]
            - targets[label]
        ),
        reverse=True,
    )

    for label in fractions[:remaining]:
        targets[label] += 1

    # --------------------------------------------------------
    # Sélection
    # --------------------------------------------------------

    selected = []

    for label in labels:

        group = groups[label].copy()

        rng.shuffle(group)

        selected.extend(
            group[
                :targets[label]
            ]
        )

    rng.shuffle(selected)

    return selected


# ============================================================
# Conversion finale DPO
# ============================================================

def convert_dpo_row(
    row,
    split,
):
    label_type = row.get("label_type")

    if label_type not in DPO_LABELS:
        raise RuntimeError(
            f"label_type DPO invalide : {label_type!r}"
        )

    return {
        "id": stable_id(
            f"dpo_{split}",
            row,
        ),
        "prompt": row.get("prompt"),
        "chosen": row.get("chosen"),
        "rejected": row.get("rejected"),
        "source": {
            "dataset": "UltraMedical-Preference",
            "split": row.get(
                "_original_split"
            ),
            "label_type": label_type,
        },
        "quality": {
            "preference_valid": True,
            "clinically_validated": False,
            "clinical_validation_status": (
                "pending"
            ),
        },
        "provenance": {
            "source_id": source_id(row),
            "processing_version": (
                PROCESSING_VERSION
            ),
            "dpo_split": split,
        },
    }


# ============================================================
# Contrôles build
# ============================================================

def get_prompt_set(rows):
    return {
        prompt_key(row.get("prompt"))
        for row in rows
        if prompt_key(row.get("prompt"))
    }


def check_output_ids(rows, dataset_name):
    ids = [
        row.get("id")
        for row in rows
    ]

    invalid = [
        value
        for value in ids
        if not isinstance(value, str)
        or not value.strip()
    ]

    if invalid:
        raise RuntimeError(
            f"{dataset_name}: "
            f"{len(invalid)} ID invalides."
        )

    counts = Counter(ids)

    duplicates = {
        value: count
        for value, count in counts.items()
        if count > 1
    }

    if duplicates:
        raise RuntimeError(
            f"{dataset_name}: "
            f"IDs dupliqués : {duplicates}"
        )


def check_no_leakage(
    dpo_train,
    dpo_dev,
    rewardbench,
    sft_rows,
):
    train_prompts = get_prompt_set(
        dpo_train
    )

    dev_prompts = get_prompt_set(
        dpo_dev
    )

    reward_prompts = get_prompt_set(
        rewardbench
    )

    sft_prompts = {
        normalize_text(
            row.get("instruction")
        )
        for row in sft_rows
        if normalize_text(
            row.get("instruction")
        )
    }

    intersections = {
        "SFT → DPO train": (
            sft_prompts & train_prompts
        ),
        "SFT → DPO validation": (
            sft_prompts & dev_prompts
        ),
        "DPO train ↔ validation": (
            train_prompts & dev_prompts
        ),
        "DPO train → RewardBench": (
            train_prompts & reward_prompts
        ),
        "DPO validation → RewardBench": (
            dev_prompts & reward_prompts
        ),
    }

    print("\nLeakage checks:")

    for name, values in intersections.items():
        print(
            f"  {name:<30}: {len(values)}"
        )

    leaked = {
        name: values
        for name, values in intersections.items()
        if values
    }

    if leaked:
        details = ", ".join(
            f"{name}={len(values)}"
            for name, values in leaked.items()
        )

        raise RuntimeError(
            "Leakage détecté avant écriture : "
            + details
        )


def check_exact_duplicate_pairs(
    rows,
    dataset_name,
):
    seen = set()

    for row in rows:
        key = preference_key(row)

        if key in seen:
            raise RuntimeError(
                f"{dataset_name}: "
                "doublon exact de préférence détecté."
            )

        seen.add(key)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build DPO train/validation datasets "
            "and Medical RewardBench."
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
        "--sft",
        type=Path,
        default=DEFAULT_SFT,
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--dev-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir
    processed_dir = args.processed_dir

    train_path = (
        raw_dir / "train.json"
    )

    dev_path = (
        raw_dir / "dev.json"
    )

    test_path = (
        raw_dir / "test.json"
    )

    output_train = (
        processed_dir
        / "dpo_train.jsonl"
    )

    output_dev = (
        processed_dir
        / "dpo_validation.jsonl"
    )

    output_rewardbench = (
        processed_dir
        / "medical_rewardbench.jsonl"
    )

    # ========================================================
    # 1. Chargement
    # ========================================================

    print("=" * 70)
    print("BUILD DPO DATASETS")
    print("=" * 70)

    print(
        "\n[1/6] Loading source datasets"
    )

    train_rows = load_preference_rows(
        train_path
    )

    dev_rows = load_preference_rows(
        dev_path
    )

    test_rows = load_preference_rows(
        test_path
    )

    print(
        f"UltraMedical train : "
        f"{len(train_rows)}"
    )

    print(
        f"UltraMedical dev   : "
        f"{len(dev_rows)}"
    )

    print(
        f"UltraMedical test  : "
        f"{len(test_rows)}"
    )

    # ========================================================
    # 2. SFT
    # ========================================================

    print(
        "\n[2/6] Loading SFT prompts"
    )

    sft_rows = load_jsonl(
        args.sft
    )

    sft_prompts = {
        normalize_text(
            row.get("instruction")
        )
        for row in sft_rows
        if normalize_text(
            row.get("instruction")
        )
    }

    print(
        f"SFT prompts : "
        f"{len(sft_prompts)}"
    )

    # ========================================================
    # 3. RewardBench
    # ========================================================

    print(
        "\n[3/6] Building RewardBench"
    )

    rewardbench = build_rewardbench(
        test_rows
    )

    rewardbench_prompts = get_prompt_set(
        rewardbench
    )

    print(
        f"RewardBench final : "
        f"{len(rewardbench)}"
    )

    # ========================================================
    # 4. Préparation DPO
    # ========================================================

    print(
        "\n[4/6] Preparing DPO candidates"
    )

    excluded_prompts = (
        sft_prompts
        | rewardbench_prompts
    )

    for row in train_rows:
        row["_original_split"] = "train"

    for row in dev_rows:
        row["_original_split"] = "dev"

    (
        clean_dev,
        dev_no_label,
        dev_invalid_label,
        dev_invalid_content,
        dev_excluded,
        dev_duplicates,
    ) = clean_candidates(
        dev_rows,
        excluded_prompts,
    )

    (
        clean_train,
        train_no_label,
        train_invalid_label,
        train_invalid_content,
        train_excluded,
        train_duplicates,
    ) = clean_candidates(
        train_rows,
        excluded_prompts,
    )

    print(
        f"DPO dev candidates   : "
        f"{len(clean_dev)}"
    )

    print(
        f"DPO train candidates : "
        f"{len(clean_train)}"
    )

    if dev_no_label:
        print(
            f"Dev ignorés sans label_type : "
            f"{dev_no_label}"
        )

    if train_no_label:
        print(
            f"Train ignorés sans label_type : "
            f"{train_no_label}"
        )

    if dev_invalid_label:
        print(
            f"Dev ignorés avec label_type "
            f"invalide : {dev_invalid_label}"
        )

    if train_invalid_label:
        print(
            f"Train ignorés avec label_type "
            f"invalide : {train_invalid_label}"
        )

    if dev_invalid_content:
        print(
            f"Dev ignorés contenu invalide : "
            f"{dev_invalid_content}"
        )

    if train_invalid_content:
        print(
            f"Train ignorés contenu invalide : "
            f"{train_invalid_content}"
        )

    if dev_excluded:
        print(
            f"Dev exclus pour leakage : "
            f"{dev_excluded}"
        )

    if train_excluded:
        print(
            f"Train exclus pour leakage : "
            f"{train_excluded}"
        )

    if dev_duplicates:
        print(
            f"Doublons dev ignorés : "
            f"{dev_duplicates}"
        )

    if train_duplicates:
        print(
            f"Doublons train ignorés : "
            f"{train_duplicates}"
        )

    # ========================================================
    # 5. Sampling
    # ========================================================

    print(
        "\n[5/6] Sampling DPO train / validation"
    )

    # --------------------------------------------------------
    # Validation d'abord
    # --------------------------------------------------------

    dpo_dev_raw = stratified_sample(
        clean_dev,
        args.dev_size,
        seed=args.seed,
    )

    dev_prompts = {
        prompt_key(row["prompt"])
        for row in dpo_dev_raw
    }

    # --------------------------------------------------------
    # Suppression des prompts validation du train
    # --------------------------------------------------------

    clean_train = [
        row
        for row in clean_train
        if prompt_key(
            row["prompt"]
        )
        not in dev_prompts
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    dpo_train_raw = stratified_sample(
        clean_train,
        args.train_size,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Conversion finale
    # --------------------------------------------------------

    dpo_train = [
        convert_dpo_row(
            row,
            "train",
        )
        for row in dpo_train_raw
    ]

    dpo_dev = [
        convert_dpo_row(
            row,
            "validation",
        )
        for row in dpo_dev_raw
    ]

    # ========================================================
    # Contrôles avant écriture
    # ========================================================

    print(
        "\nPre-write checks"
    )

    # --------------------------------------------------------
    # Taille
    # --------------------------------------------------------

    if len(dpo_train) != args.train_size:
        raise RuntimeError(
            f"DPO train : "
            f"{len(dpo_train)} obtenus, "
            f"{args.train_size} attendus."
        )

    if len(dpo_dev) != args.dev_size:
        raise RuntimeError(
            f"DPO validation : "
            f"{len(dpo_dev)} obtenus, "
            f"{args.dev_size} attendus."
        )

    if len(rewardbench) != 776:
        raise RuntimeError(
            f"RewardBench : "
            f"{len(rewardbench)} obtenus, "
            "776 attendus."
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    for dataset_name, rows in (
        ("DPO train", dpo_train),
        ("DPO validation", dpo_dev),
    ):

        invalid_labels = [
            row.get("source", {}).get(
                "label_type"
            )
            for row in rows
            if row.get("source", {}).get(
                "label_type"
            ) not in DPO_LABELS
        ]

        if invalid_labels:
            raise RuntimeError(
                f"{dataset_name} contient des "
                f"label_type invalides : "
                f"{Counter(invalid_labels)}"
            )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    check_output_ids(
        dpo_train,
        "DPO train",
    )

    check_output_ids(
        dpo_dev,
        "DPO validation",
    )

    check_output_ids(
        rewardbench,
        "RewardBench",
    )

    # --------------------------------------------------------
    # Doublons exacts
    # --------------------------------------------------------

    check_exact_duplicate_pairs(
        dpo_train,
        "DPO train",
    )

    check_exact_duplicate_pairs(
        dpo_dev,
        "DPO validation",
    )

    check_exact_duplicate_pairs(
        rewardbench,
        "RewardBench",
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    check_no_leakage(
        dpo_train,
        dpo_dev,
        rewardbench,
        sft_rows,
    )

    # ========================================================
    # Écriture
    # ========================================================

    write_jsonl(
        output_train,
        dpo_train,
    )

    write_jsonl(
        output_dev,
        dpo_dev,
    )

    write_jsonl(
        output_rewardbench,
        rewardbench,
    )

    # ========================================================
    # 6. Contrôles finaux
    # ========================================================

    print(
        "\n[6/6] Final checks"
    )

    train_prompts = get_prompt_set(
        dpo_train
    )

    dev_prompts = get_prompt_set(
        dpo_dev
    )

    rb_prompts = get_prompt_set(
        rewardbench
    )

    print(
        f"DPO train       : "
        f"{len(dpo_train)}"
    )

    print(
        f"DPO validation  : "
        f"{len(dpo_dev)}"
    )

    print(
        f"RewardBench     : "
        f"{len(rewardbench)}"
    )

    print(
        f"SFT → train     : "
        f"{len(sft_prompts & train_prompts)}"
    )

    print(
        f"SFT → validation: "
        f"{len(sft_prompts & dev_prompts)}"
    )

    print(
        f"train ↔ dev     : "
        f"{len(train_prompts & dev_prompts)}"
    )

    print(
        f"train → RB      : "
        f"{len(train_prompts & rb_prompts)}"
    )

    print(
        f"validation → RB : "
        f"{len(dev_prompts & rb_prompts)}"
    )

    # --------------------------------------------------------
    # Distribution train
    # --------------------------------------------------------

    print(
        "\nDPO train distribution:"
    )

    for label in sorted(DPO_LABELS):

        count = sum(
            1
            for row in dpo_train
            if row["source"]["label_type"]
            == label
        )

        percentage = (
            count
            / len(dpo_train)
            * 100
            if dpo_train
            else 0
        )

        print(
            f"  {label:<8} "
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Distribution validation
    # --------------------------------------------------------

    print(
        "\nDPO validation distribution:"
    )

    for label in sorted(DPO_LABELS):

        count = sum(
            1
            for row in dpo_dev
            if row["source"]["label_type"]
            == label
        )

        percentage = (
            count
            / len(dpo_dev)
            * 100
            if dpo_dev
            else 0
        )

        print(
            f"  {label:<8} "
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Fichiers
    # --------------------------------------------------------

    print(
        "\nFiles generated:"
    )

    print(
        f"  {output_train}"
    )

    print(
        f"  {output_dev}"
    )

    print(
        f"  {output_rewardbench}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BUILD COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
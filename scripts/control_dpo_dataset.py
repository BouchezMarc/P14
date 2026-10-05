from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_TRAIN = Path(
    "data/processed/dpo_train.jsonl"
)

DEFAULT_DEV = Path(
    "data/processed/dpo_validation.jsonl"
)

DEFAULT_REWARDBENCH = Path(
    "data/processed/medical_rewardbench.jsonl"
)

DEFAULT_SFT = Path(
    "data/processed/sft_5000.jsonl"
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REQUIRED_DPO_FIELDS = {
    "id",
    "prompt",
    "chosen",
    "rejected",
    "source",
    "quality",
    "provenance",
}

REQUIRED_SOURCE_FIELDS = {
    "dataset",
    "split",
    "label_type",
}

REQUIRED_QUALITY_FIELDS = {
    "preference_valid",
    "clinically_validated",
    "clinical_validation_status",
}

VALID_LABELS = {
    "easy",
    "hard",
    "length",
}

VALID_REWARDBENCH_LABELS = {
    "easy",
    "hard",
    "length",
    "human",
}

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d[\d .()-]{7,}\d)"
    r"(?!\d)"
)

URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """
    Load JSONL and return:
        records
        errors
    """

    records = []
    errors = []

    if not path.exists():
        errors.append(
            f"File not found: {path}"
        )
        return records, errors

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                errors.append(
                    f"{path}:{line_no}: "
                    f"invalid JSON: {exc}"
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    f"{path}:{line_no}: "
                    f"record is not an object"
                )
                continue

            records.append(record)

    return records, errors


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_text(value) -> str:
    """
    Normalize plain text for comparisons.
    """

    if not isinstance(value, str):
        return ""

    value = value.lower().strip()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def normalize_conversation(value) -> str:
    """
    Normalize a conversation stored as:
        [
            {"role": "...", "content": "..."},
            ...
        ]

    Used only for comparisons.
    """

    if not isinstance(value, list):
        return ""

    normalized_messages = []

    for message in value:

        if not isinstance(message, dict):
            continue

        role = normalize_text(
            message.get("role")
        )

        content = normalize_text(
            message.get("content")
        )

        if role or content:
            normalized_messages.append(
                f"{role}:{content}"
            )

    return "\n".join(
        normalized_messages
    )


def normalize_preference_value(value) -> str:
    """
    Normalize either:
        - plain string
        - conversation list
    """

    if isinstance(value, str):
        return normalize_text(value)

    if isinstance(value, list):
        return normalize_conversation(value)

    return ""


def preference_key(record: dict):

    return (
        normalize_preference_value(
            record.get("prompt")
        ),
        normalize_preference_value(
            record.get("chosen")
        ),
        normalize_preference_value(
            record.get("rejected")
        ),
    )


# ---------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------

def detect_pii(
    records: list[dict],
) -> list[tuple[int, str]]:

    findings = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        text = json.dumps(
            record,
            ensure_ascii=False,
        )

        if EMAIL_RE.search(text):
            findings.append(
                (index, "email")
            )

        if PHONE_RE.search(text):
            findings.append(
                (index, "phone")
            )

        if URL_RE.search(text):
            findings.append(
                (index, "url")
            )

    return findings


# ---------------------------------------------------------------------
# Basic structure checks
# ---------------------------------------------------------------------

def check_required_fields(
    records: list[dict],
    errors: list[str],
    dataset_name: str,
):

    for index, record in enumerate(
        records,
        start=1,
    ):

        missing = (
            REQUIRED_DPO_FIELDS
            - set(record.keys())
        )

        if missing:
            errors.append(
                f"{dataset_name}[{index}] "
                f"missing fields: "
                f"{sorted(missing)}"
            )

        source = record.get("source")

        if not isinstance(source, dict):

            errors.append(
                f"{dataset_name}[{index}] "
                f"source must be an object"
            )

        else:

            missing_source = (
                REQUIRED_SOURCE_FIELDS
                - set(source.keys())
            )

            if missing_source:

                errors.append(
                    f"{dataset_name}[{index}] "
                    f"source missing: "
                    f"{sorted(missing_source)}"
                )

        quality = record.get("quality")

        if not isinstance(quality, dict):

            errors.append(
                f"{dataset_name}[{index}] "
                f"quality must be an object"
            )

        else:

            missing_quality = (
                REQUIRED_QUALITY_FIELDS
                - set(quality.keys())
            )

            if missing_quality:

                errors.append(
                    f"{dataset_name}[{index}] "
                    f"quality missing: "
                    f"{sorted(missing_quality)}"
                )


# ---------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------

def valid_conversation(
    value,
) -> bool:
    """
    Validate a conversation represented by:
        [
            {"role": "...", "content": "..."},
            ...
        ]
    """

    if not isinstance(value, list):
        return False

    if not value:
        return False

    valid_message_found = False

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

        valid_message_found = True

    return valid_message_found


def valid_preference_value(
    value,
) -> bool:
    """
    Accept either:
        - non-empty string
        - non-empty conversation list
    """

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return valid_conversation(value)

    return False


# ---------------------------------------------------------------------
# Content checks
# ---------------------------------------------------------------------

def check_content(
    records: list[dict],
    errors: list[str],
    warnings: list[str],
    dataset_name: str,
):

    for index, record in enumerate(
        records,
        start=1,
    ):

        prompt = record.get(
            "prompt",
            "",
        )

        chosen = record.get(
            "chosen",
            "",
        )

        rejected = record.get(
            "rejected",
            "",
        )

        # -------------------------------------------------------------
        # Prompt
        # -------------------------------------------------------------

        if (
            not isinstance(prompt, str)
            or not prompt.strip()
        ):

            errors.append(
                f"{dataset_name}[{index}] "
                f"empty prompt"
            )

        # -------------------------------------------------------------
        # Chosen
        # -------------------------------------------------------------

        if not valid_preference_value(
            chosen
        ):

            errors.append(
                f"{dataset_name}[{index}] "
                f"empty/invalid chosen"
            )

        # -------------------------------------------------------------
        # Rejected
        # -------------------------------------------------------------

        if not valid_preference_value(
            rejected
        ):

            errors.append(
                f"{dataset_name}[{index}] "
                f"empty/invalid rejected"
            )

        # -------------------------------------------------------------
        # Chosen == rejected
        # -------------------------------------------------------------

        chosen_normalized = (
            normalize_preference_value(
                chosen
            )
        )

        rejected_normalized = (
            normalize_preference_value(
                rejected
            )
        )

        if (
            chosen_normalized
            and rejected_normalized
            and chosen_normalized
            == rejected_normalized
        ):

            errors.append(
                f"{dataset_name}[{index}] "
                f"chosen == rejected"
            )


# ---------------------------------------------------------------------
# ID checks
# ---------------------------------------------------------------------

def check_ids(
    records: list[dict],
    errors: list[str],
    dataset_name: str,
):

    ids = []

    for record in records:

        value = record.get("id")

        if (
            not isinstance(value, str)
            or not value
        ):

            errors.append(
                f"{dataset_name}: "
                f"missing/invalid id"
            )

            continue

        ids.append(value)

    counts = Counter(ids)

    duplicates = {
        value: count
        for value, count in counts.items()
        if count > 1
    }

    for value, count in duplicates.items():

        errors.append(
            f"{dataset_name}: "
            f"duplicate id '{value}' "
            f"({count} occurrences)"
        )


# ---------------------------------------------------------------------
# Duplicate checks
# ---------------------------------------------------------------------

def check_duplicates(
    records: list[dict],
    errors: list[str],
    warnings: list[str],
    dataset_name: str,
):

    prompts = Counter()
    pairs = Counter()

    for record in records:

        prompt = normalize_text(
            record.get("prompt")
        )

        if prompt:
            prompts[prompt] += 1

        key = preference_key(record)

        if all(key):
            pairs[key] += 1

    duplicate_prompts = [
        (prompt, count)
        for prompt, count in prompts.items()
        if count > 1
    ]

    duplicate_pairs = [
        (pair, count)
        for pair, count in pairs.items()
        if count > 1
    ]

    if duplicate_prompts:

        warnings.append(
            f"{dataset_name}: "
            f"{len(duplicate_prompts)} "
            f"repeated prompts"
        )

    if duplicate_pairs:

        errors.append(
            f"{dataset_name}: "
            f"{len(duplicate_pairs)} "
            f"duplicate preference pairs"
        )


# ---------------------------------------------------------------------
# DPO metadata
# ---------------------------------------------------------------------

def check_dpo_metadata(
    records: list[dict],
    expected_split: str,
    errors: list[str],
    dataset_name: str,
):

    for index, record in enumerate(
        records,
        start=1,
    ):

        source = record.get(
            "source",
            {},
        )

        if not isinstance(
            source,
            dict,
        ):
            continue

        if (
            source.get("dataset")
            != "UltraMedical-Preference"
        ):

            errors.append(
                f"{dataset_name}[{index}]: "
                f"invalid source dataset"
            )

        if (
            source.get("split")
            != expected_split
        ):

            errors.append(
                f"{dataset_name}[{index}]: "
                f"expected split "
                f"'{expected_split}', "
                f"got "
                f"'{source.get('split')}'"
            )

        label = source.get(
            "label_type"
        )

        if label not in VALID_LABELS:

            errors.append(
                f"{dataset_name}[{index}]: "
                f"invalid label_type "
                f"'{label}'"
            )

        quality = record.get(
            "quality",
            {},
        )

        if (
            quality.get(
                "clinically_validated"
            )
            is not False
        ):

            errors.append(
                f"{dataset_name}[{index}]: "
                f"clinically_validated "
                f"must be false"
            )

        if (
            quality.get(
                "clinical_validation_status"
            )
            != "pending"
        ):

            errors.append(
                f"{dataset_name}[{index}]: "
                f"clinical_validation_status "
                f"must be 'pending'"
            )


# ---------------------------------------------------------------------
# RewardBench checks
# ---------------------------------------------------------------------

def check_rewardbench(
    records: list[dict],
    errors: list[str],
    warnings: list[str],
):

    # ---------------------------------------------------------------
    # Count
    # ---------------------------------------------------------------

    if len(records) != 776:

        errors.append(
            "RewardBench: expected 776 records, "
            f"got {len(records)}"
        )

    # ---------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------

    labels = Counter(
        record.get("label_type")
        for record in records
    )

    expected = {
        "easy": 237,
        "hard": 196,
        "length": 180,
        "human": 163,
    }

    for label, expected_count in expected.items():

        actual = labels.get(
            label,
            0,
        )

        if actual != expected_count:

            errors.append(
                "RewardBench: "
                f"{label} expected "
                f"{expected_count}, "
                f"got {actual}"
            )

    unexpected = (
        set(labels)
        - VALID_REWARDBENCH_LABELS
    )

    if unexpected:

        errors.append(
            "RewardBench: "
            f"unexpected labels: "
            f"{sorted(unexpected)}"
        )

    # ---------------------------------------------------------------
    # Record-level checks
    # ---------------------------------------------------------------

    for index, record in enumerate(
        records,
        start=1,
    ):

        source = record.get(
            "source",
            {},
        )

        if not isinstance(
            source,
            dict,
        ):

            errors.append(
                f"RewardBench[{index}]: "
                f"source must be an object"
            )

        else:

            if (
                source.get("dataset")
                != "UltraMedical-Preference"
            ):

                errors.append(
                    f"RewardBench[{index}]: "
                    f"invalid source dataset"
                )

            if (
                source.get("split")
                != "test"
            ):

                errors.append(
                    f"RewardBench[{index}]: "
                    f"source split is not test"
                )

        quality = record.get(
            "quality",
            {},
        )

        if not isinstance(
            quality,
            dict,
        ):

            errors.append(
                f"RewardBench[{index}]: "
                f"quality must be an object"
            )

            continue

        if (
            quality.get(
                "expert_reviewed_labels"
            )
            is not True
        ):

            errors.append(
                f"RewardBench[{index}]: "
                f"expert_reviewed_labels "
                f"is not true"
            )

        if (
            quality.get(
                "clinical_validation_status"
            )
            != "expert_reviewed_labels"
        ):

            errors.append(
                f"RewardBench[{index}]: "
                f"invalid clinical validation "
                f"status"
            )

    return labels


# ---------------------------------------------------------------------
# Cross-dataset leakage
# ---------------------------------------------------------------------

def prompt_set(
    records: list[dict],
) -> set[str]:

    return {
        normalize_text(
            record.get("prompt")
        )
        for record in records
        if normalize_text(
            record.get("prompt")
        )
    }


def check_leakage(
    train: list[dict],
    dev: list[dict],
    rewardbench: list[dict],
    sft: list[dict],
    errors: list[str],
):

    train_prompts = prompt_set(
        train
    )

    dev_prompts = prompt_set(
        dev
    )

    reward_prompts = prompt_set(
        rewardbench
    )

    sft_prompts = {
        normalize_text(
            record.get("instruction")
        )
        for record in sft
        if normalize_text(
            record.get("instruction")
        )
    }

    train_sft = (
        train_prompts
        & sft_prompts
    )

    dev_sft = (
        dev_prompts
        & sft_prompts
    )

    train_dev = (
        train_prompts
        & dev_prompts
    )

    train_reward = (
        train_prompts
        & reward_prompts
    )

    dev_reward = (
        dev_prompts
        & reward_prompts
    )

    if train_sft:

        errors.append(
            "SFT → DPO train leakage: "
            f"{len(train_sft)} prompts"
        )

    if dev_sft:

        errors.append(
            "SFT → DPO validation leakage: "
            f"{len(dev_sft)} prompts"
        )

    if train_dev:

        errors.append(
            "DPO train → validation leakage: "
            f"{len(train_dev)} prompts"
        )

    if train_reward:

        errors.append(
            "DPO train → RewardBench leakage: "
            f"{len(train_reward)} prompts"
        )

    if dev_reward:

        errors.append(
            "DPO validation → RewardBench leakage: "
            f"{len(dev_reward)} prompts"
        )

    return {
        "sft_train": len(train_sft),
        "sft_dev": len(dev_sft),
        "train_dev": len(train_dev),
        "train_rewardbench": len(train_reward),
        "dev_rewardbench": len(dev_reward),
    }


# ---------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------

def print_distribution(
    name: str,
    records: list[dict],
):

    labels = Counter(
        record.get("source", {}).get(
            "label_type"
        )
        for record in records
    )

    print(f"\n{name}")
    print("-" * 50)

    total = len(records)

    for label in sorted(labels):

        count = labels[label]

        percentage = (
            100 * count / total
            if total
            else 0
        )

        print(
            f"{label:<10} "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Validate DPO datasets."
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN,
    )

    parser.add_argument(
        "--dev",
        type=Path,
        default=DEFAULT_DEV,
    )

    parser.add_argument(
        "--rewardbench",
        type=Path,
        default=DEFAULT_REWARDBENCH,
    )

    parser.add_argument(
        "--sft",
        type=Path,
        default=DEFAULT_SFT,
    )

    args = parser.parse_args()

    errors = []
    warnings = []

    print("=" * 70)
    print("DPO DATASET VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    print("\n[1/7] Loading files")

    train, train_load_errors = load_jsonl(
        args.train
    )

    dev, dev_load_errors = load_jsonl(
        args.dev
    )

    rewardbench, rewardbench_load_errors = (
        load_jsonl(
            args.rewardbench
        )
    )

    sft, sft_load_errors = load_jsonl(
        args.sft
    )

    errors.extend(
        train_load_errors
    )

    errors.extend(
        dev_load_errors
    )

    errors.extend(
        rewardbench_load_errors
    )

    errors.extend(
        sft_load_errors
    )

    print(
        f"DPO train       : {len(train)}"
    )

    print(
        f"DPO validation  : {len(dev)}"
    )

    print(
        f"RewardBench     : {len(rewardbench)}"
    )

    print(
        f"SFT             : {len(sft)}"
    )

    # ---------------------------------------------------------------
    # Structure
    # ---------------------------------------------------------------

    print("\n[2/7] Structural checks")

    # DPO only:
    check_required_fields(
        train,
        errors,
        "dpo_train",
    )

    check_required_fields(
        dev,
        errors,
        "dpo_validation",
    )

    # IDs
    check_ids(
        train,
        errors,
        "dpo_train",
    )

    check_ids(
        dev,
        errors,
        "dpo_validation",
    )

    check_ids(
        rewardbench,
        errors,
        "rewardbench",
    )

    # Content
    check_content(
        train,
        errors,
        warnings,
        "dpo_train",
    )

    check_content(
        dev,
        errors,
        warnings,
        "dpo_validation",
    )

    check_content(
        rewardbench,
        errors,
        warnings,
        "rewardbench",
    )

    # ---------------------------------------------------------------
    # DPO metadata
    # ---------------------------------------------------------------

    print("\n[3/7] DPO metadata")

    check_dpo_metadata(
        train,
        "train",
        errors,
        "dpo_train",
    )

    check_dpo_metadata(
        dev,
        "dev",
        errors,
        "dpo_validation",
    )

    # ---------------------------------------------------------------
    # Duplicates
    # ---------------------------------------------------------------

    print("\n[4/7] Duplicate checks")

    check_duplicates(
        train,
        errors,
        warnings,
        "dpo_train",
    )

    check_duplicates(
        dev,
        errors,
        warnings,
        "dpo_validation",
    )

    check_duplicates(
        rewardbench,
        errors,
        warnings,
        "rewardbench",
    )

    # ---------------------------------------------------------------
    # RewardBench
    # ---------------------------------------------------------------

    print("\n[5/7] RewardBench checks")

    reward_labels = check_rewardbench(
        rewardbench,
        errors,
        warnings,
    )

    # ---------------------------------------------------------------
    # Leakage
    # ---------------------------------------------------------------

    print("\n[6/7] Leakage checks")

    leakage = check_leakage(
        train,
        dev,
        rewardbench,
        sft,
        errors,
    )

    # ---------------------------------------------------------------
    # PII
    # ---------------------------------------------------------------

    print("\n[7/7] Basic PII checks")

    pii_train = detect_pii(
        train
    )

    pii_dev = detect_pii(
        dev
    )

    pii_reward = detect_pii(
        rewardbench
    )

    if pii_train:

        warnings.append(
            f"DPO train: "
            f"{len(pii_train)} records "
            f"with possible PII"
        )

    if pii_dev:

        warnings.append(
            f"DPO validation: "
            f"{len(pii_dev)} records "
            f"with possible PII"
        )

    if pii_reward:

        warnings.append(
            f"RewardBench: "
            f"{len(pii_reward)} records "
            f"with possible PII"
        )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    print_distribution(
        "DPO train distribution",
        train,
    )

    print_distribution(
        "DPO validation distribution",
        dev,
    )

    print(
        "\nRewardBench distribution"
    )

    for label in sorted(
        reward_labels
    ):

        print(
            f"  {label:<10}: "
            f"{reward_labels[label]}"
        )

    print(
        "\nLeakage"
    )

    print(
        f"  SFT → train       : "
        f"{leakage['sft_train']}"
    )

    print(
        f"  SFT → validation  : "
        f"{leakage['sft_dev']}"
    )

    print(
        f"  train ↔ validation: "
        f"{leakage['train_dev']}"
    )

    print(
        f"  train → RewardBench: "
        f"{leakage['train_rewardbench']}"
    )

    print(
        f"  validation → RewardBench: "
        f"{leakage['dev_rewardbench']}"
    )

    # ---------------------------------------------------------------
    # Result
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    if errors:

        print(
            f"❌ FAILED — "
            f"{len(errors)} error(s)"
        )

        for error in errors:

            print(
                f"  ERROR: {error}"
            )

    else:

        print(
            "✅ TECHNICAL CHECKS PASSED"
        )

    if warnings:

        print(
            f"\n⚠️ {len(warnings)} warning(s)"
        )

        for warning in warnings:

            print(
                f"  WARNING: {warning}"
            )

    print("=" * 70)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""
report_pii.py

Analyse les alertes PII présentes dans les datasets DPO/RewardBench
sans modifier les fichiers source.

Entrées :
    data/processed/dpo_train.jsonl
    data/processed/dpo_validation.jsonl
    data/processed/medical_rewardbench.jsonl

Sorties :
    reports/pii_report.json
    reports/pii_report.txt

Important :
    - Les champs techniques sont exclus du scan PII.
    - Les regex produisent des ALERTES, pas une preuve de PII.
    - Aucun dataset source n'est modifié.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("reports")

DATASETS = {
    "dpo_train": "dpo_train.jsonl",
    "dpo_validation": "dpo_validation.jsonl",
    "rewardbench": "medical_rewardbench.jsonl",
}


# ============================================================
# CHAMPS À EXCLURE DU SCAN
# ============================================================

# Ces champs sont des identifiants techniques et ne doivent pas
# être interprétés comme des données personnelles.

EXCLUDED_FIELDS = {
    "id",
    "provenance.source_id",
    "provenance.processing_version",
}


# ============================================================
# REGEX PII
# ============================================================

PII_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        re.IGNORECASE,
    ),

    "phone_international": re.compile(
        r"(?<!\w)"
        r"(?:\+|00)"
        r"\d{1,3}"
        r"(?:[\s().-]*\d{2,4}){2,6}"
        r"(?!\w)"
    ),

    "phone_french": re.compile(
        r"(?<!\d)"
        r"(?:0[1-9])"
        r"(?:[\s.-]?\d{2}){4}"
        r"(?!\d)"
    ),

    "ip_address": re.compile(
        r"\b(?:"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"\."
        r"){3}"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"\b"
    ),

    "url": re.compile(
        r"\b(?:https?://|www\.)"
        r"[^\s<>\"]+",
        re.IGNORECASE,
    ),

    "date_numeric": re.compile(
        r"(?<!\d)"
        r"(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|"
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r")"
        r"(?!\d)"
    ),

    "date_month_year": re.compile(
        r"\b"
        r"(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
        r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?"
        r")"
        r"\s+\d{4}"
        r"\b",
        re.IGNORECASE,
    ),

    "social_security_like": re.compile(
        r"(?<!\d)"
        r"\d{3}[-\s]\d{2}[-\s]\d{4}"
        r"(?!\d)"
    ),

    "us_zip_code": re.compile(
        r"(?<!\d)"
        r"\d{5}(?:-\d{4})?"
        r"(?!\d)"
    ),

    "french_postal_code": re.compile(
        r"(?<!\d)"
        r"(?:0[1-9]|[1-8]\d|9[0-5]|2A|2B)"
        r"\s?\d{3}"
        r"(?!\d)"
    ),
}


# ============================================================
# CHARGEMENT JSONL
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Charge un fichier JSONL."""

    records = []

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"JSON invalide dans {path}, "
                    f"ligne {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):

                raise ValueError(
                    f"Record non-dict dans {path}, "
                    f"ligne {line_number}"
                )

            record["_report_line_number"] = line_number

            records.append(record)

    return records


# ============================================================
# ID DU RECORD
# ============================================================

def safe_record_id(
    record: dict[str, Any],
) -> str:
    """
    Récupère l'identifiant technique du record.

    Cet ID est utilisé pour le rapport mais n'est jamais scanné
    comme contenu PII.
    """

    value = record.get("id")

    if value is not None:
        return str(value)

    provenance = record.get("provenance")

    if isinstance(provenance, dict):

        source_id = provenance.get("source_id")

        if source_id is not None:
            return str(source_id)

    return "<NO_ID>"


# ============================================================
# WALK DES CHAMPS
# ============================================================

def walk_scannable_fields(
    value: Any,
    path: str = "",
):
    """
    Parcourt récursivement le JSON et retourne uniquement
    les champs textuels qui doivent être analysés.

    Les champs techniques EXCLUDED_FIELDS sont ignorés.
    """

    if isinstance(value, str):

        if path not in EXCLUDED_FIELDS:

            yield path, value

        return

    if isinstance(value, dict):

        for key, child in value.items():

            if key == "_report_line_number":
                continue

            child_path = (
                f"{path}.{key}"
                if path
                else key
            )

            # Exclusion directe des champs techniques.
            if child_path in EXCLUDED_FIELDS:
                continue

            yield from walk_scannable_fields(
                child,
                child_path,
            )

        return

    if isinstance(value, list):

        for index, child in enumerate(value):

            child_path = f"{path}[{index}]"

            yield from walk_scannable_fields(
                child,
                child_path,
            )


# ============================================================
# ANALYSE D'UN RECORD
# ============================================================

def find_field_matches(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Recherche les alertes PII dans les champs autorisés.
    """

    matches = []

    for field_path, value in walk_scannable_fields(record):

        for pii_type, pattern in PII_PATTERNS.items():

            for match in pattern.finditer(value):

                matched_text = match.group(0)

                start = match.start()
                end = match.end()

                context_start = max(
                    0,
                    start - 80,
                )

                context_end = min(
                    len(value),
                    end + 80,
                )

                context = value[
                    context_start:context_end
                ]

                relative_start = (
                    start - context_start
                )

                relative_end = (
                    end - context_start
                )

                masked_context = (
                    context[:relative_start]
                    + "[PII]"
                    + context[relative_end:]
                )

                matches.append(
                    {
                        "type": pii_type,
                        "field": field_path,
                        "match": matched_text,
                        "context": masked_context,
                        "start": start,
                        "end": end,
                    }
                )

    return matches


# ============================================================
# ANALYSE DATASET
# ============================================================

def analyse_dataset(
    dataset_name: str,
    path: Path,
) -> dict[str, Any]:

    records = load_jsonl(path)

    type_counter = Counter()
    field_counter = Counter()

    # Nombre de records uniques par type.
    unique_records_by_type = {
        pii_type: set()
        for pii_type in PII_PATTERNS
    }

    records_with_alerts = 0
    total_matches = 0

    alerts = []

    for record in records:

        matches = find_field_matches(record)

        if not matches:
            continue

        records_with_alerts += 1
        total_matches += len(matches)

        record_id = safe_record_id(record)

        record_types = set()

        for match in matches:

            pii_type = match["type"]

            type_counter[pii_type] += 1

            field_counter[
                match["field"]
            ] += 1

            unique_records_by_type[
                pii_type
            ].add(record_id)

            record_types.add(pii_type)

            alerts.append(
                {
                    "dataset": dataset_name,
                    "id": record_id,
                    "line": record.get(
                        "_report_line_number"
                    ),
                    "type": pii_type,
                    "field": match["field"],
                    "match": match["match"],
                    "context": match["context"],
                }
            )

    unique_record_counts = {
        pii_type: len(record_ids)
        for pii_type, record_ids
        in unique_records_by_type.items()
        if record_ids
    }

    return {
        "dataset": dataset_name,
        "file": str(path),

        "total_records": len(records),

        "records_with_alerts": records_with_alerts,

        "total_matches": total_matches,

        "types": dict(type_counter),

        "unique_records_by_type": (
            unique_record_counts
        ),

        "fields": dict(field_counter),

        "alerts": alerts,
    }


# ============================================================
# RAPPORT JSON
# ============================================================

def build_json_report(
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:

    total_records = sum(
        item["total_records"]
        for item in analyses
    )

    total_records_with_alerts = sum(
        item["records_with_alerts"]
        for item in analyses
    )

    total_matches = sum(
        item["total_matches"]
        for item in analyses
    )

    global_types = Counter()

    global_fields = Counter()

    global_unique_records_by_type = {
        pii_type: set()
        for pii_type in PII_PATTERNS
    }

    for item in analyses:

        for pii_type, count in item[
            "types"
        ].items():

            global_types[pii_type] += count

        for field, count in item[
            "fields"
        ].items():

            global_fields[field] += count

        for alert in item["alerts"]:

            pii_type = alert["type"]
            record_id = alert["id"]

            global_unique_records_by_type[
                pii_type
            ].add(
                f"{item['dataset']}::{record_id}"
            )

    global_unique_counts = {
        pii_type: len(record_ids)
        for pii_type, record_ids
        in global_unique_records_by_type.items()
        if record_ids
    }

    return {
        "report_version": "pii_report_v2",

        "description": (
            "Rapport d'analyse des alertes PII. "
            "Les champs techniques sont exclus du scan. "
            "Aucun dataset source n'est modifié."
        ),

        "excluded_fields": sorted(
            EXCLUDED_FIELDS
        ),

        "summary": {
            "total_records": total_records,
            "records_with_alerts": (
                total_records_with_alerts
            ),
            "total_matches": total_matches,
        },

        "global_types": dict(
            sorted(
                global_types.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ),

        "global_unique_records_by_type": (
            dict(
                sorted(
                    global_unique_counts.items(),
                    key=lambda item: (
                        -item[1],
                        item[0],
                    ),
                )
            )
        ),

        "global_fields": dict(
            sorted(
                global_fields.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ),

        "datasets": analyses,
    }


# ============================================================
# RAPPORT TEXTE
# ============================================================

def write_text_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:

    lines = []

    lines.append("=" * 80)
    lines.append(
        "RAPPORT PII — DPO / REWARDBENCH"
    )
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        "IMPORTANT : ce rapport ne modifie aucun dataset."
    )

    lines.append(
        "Les détections regex sont des ALERTES "
        "et peuvent contenir des faux positifs."
    )

    lines.append("")

    lines.append(
        "Champs techniques exclus du scan :"
    )

    for field in sorted(EXCLUDED_FIELDS):

        lines.append(
            f"  - {field}"
        )

    lines.append("")

    # --------------------------------------------------------
    # SYNTHÈSE
    # --------------------------------------------------------

    summary = report["summary"]

    lines.append("=== SYNTHÈSE ===")
    lines.append("")

    lines.append(
        f"Total records analysés       : "
        f"{summary['total_records']}"
    )

    lines.append(
        f"Records avec alertes PII    : "
        f"{summary['records_with_alerts']}"
    )

    lines.append(
        f"Total matchs détectés       : "
        f"{summary['total_matches']}"
    )

    lines.append("")

    # --------------------------------------------------------
    # PAR DATASET
    # --------------------------------------------------------

    lines.append("=== PAR DATASET ===")
    lines.append("")

    for dataset in report["datasets"]:

        lines.append(
            f"[{dataset['dataset']}]"
        )

        lines.append(
            f"  Records                  : "
            f"{dataset['total_records']}"
        )

        lines.append(
            f"  Records avec alertes     : "
            f"{dataset['records_with_alerts']}"
        )

        lines.append(
            f"  Matchs détectés          : "
            f"{dataset['total_matches']}"
        )

        lines.append("")

        lines.append("  Types :")

        for pii_type, count in sorted(
            dataset["types"].items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            unique_count = (
                dataset[
                    "unique_records_by_type"
                ].get(
                    pii_type,
                    0,
                )
            )

            lines.append(
                f"    - {pii_type:<25} "
                f"{count:<6} matchs | "
                f"{unique_count} records"
            )

        lines.append("")

    # --------------------------------------------------------
    # TYPES GLOBAUX
    # --------------------------------------------------------

    lines.append(
        "=== TYPES GLOBAUX ==="
    )
    lines.append("")

    global_unique = report[
        "global_unique_records_by_type"
    ]

    for pii_type, count in report[
        "global_types"
    ].items():

        unique_count = global_unique.get(
            pii_type,
            0,
        )

        lines.append(
            f"{pii_type:<30} "
            f"{count:<6} matchs | "
            f"{unique_count} records"
        )

    lines.append("")

    # --------------------------------------------------------
    # CHAMPS CONCERNÉS
    # --------------------------------------------------------

    lines.append(
        "=== CHAMPS CONCERNÉS ==="
    )
    lines.append("")

    for field, count in report[
        "global_fields"
    ].items():

        lines.append(
            f"{field:<60} {count}"
        )

    lines.append("")

    # --------------------------------------------------------
    # ALERTES DÉTAILLÉES
    # --------------------------------------------------------

    lines.append("=" * 80)
    lines.append(
        "ALERTES DÉTAILLÉES"
    )
    lines.append("=" * 80)
    lines.append("")

    for dataset in report["datasets"]:

        alerts = dataset["alerts"]

        if not alerts:
            continue

        lines.append("")
        lines.append(
            f"### {dataset['dataset']} — "
            f"{len(alerts)} matchs"
        )
        lines.append("")

        for index, alert in enumerate(
            alerts,
            start=1,
        ):

            lines.append(
                f"[{index}] ID       : "
                f"{alert['id']}"
            )

            lines.append(
                f"    Ligne      : "
                f"{alert['line']}"
            )

            lines.append(
                f"    Type       : "
                f"{alert['type']}"
            )

            lines.append(
                f"    Champ      : "
                f"{alert['field']}"
            )

            lines.append(
                f"    Match      : "
                f"{alert['match']}"
            )

            lines.append(
                f"    Contexte   : "
                f"{alert['context']}"
            )

            lines.append("")

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    lines.append("=" * 80)
    lines.append("CONCLUSION")
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        f"{summary['records_with_alerts']} records "
        "présentent au moins une alerte regex PII."
    )

    lines.append(
        f"{summary['total_matches']} matchs ont été détectés."
    )

    lines.append("")

    lines.append(
        "Les identifiants techniques ne sont pas inclus "
        "dans cette analyse."
    )

    lines.append(
        "Les codes postaux et les dates doivent être "
        "interprétés dans leur contexte avant anonymisation."
    )

    lines.append(
        "Les emails et numéros de téléphone constituent "
        "les alertes à examiner prioritairement."
    )

    lines.append("")

    lines.append(
        "Ce rapport ne constitue pas une validation "
        "juridique ou clinique de l'absence de PII."
    )

    lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyse les alertes PII des datasets "
            "DPO/RewardBench sans modifier les fichiers."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Répertoire contenant les JSONL traités."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Répertoire de sortie des rapports."
        ),
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyses = []

    print()
    print("=" * 70)
    print("RAPPORT PII")
    print("=" * 70)
    print()

    print(
        "Champs techniques exclus :"
    )

    for field in sorted(
        EXCLUDED_FIELDS
    ):

        print(
            f"  - {field}"
        )

    print()

    for dataset_name, filename in DATASETS.items():

        path = input_dir / filename

        print(
            f"Analyse {dataset_name:<20} : {path}"
        )

        if not path.exists():

            print(
                f"  [WARNING] Fichier absent : "
                f"{path}"
            )

            continue

        analysis = analyse_dataset(
            dataset_name,
            path,
        )

        analyses.append(analysis)

        print(
            f"  Records                  : "
            f"{analysis['total_records']}"
        )

        print(
            f"  Records avec alertes     : "
            f"{analysis['records_with_alerts']}"
        )

        print(
            f"  Matchs                   : "
            f"{analysis['total_matches']}"
        )

        if analysis["types"]:

            print(
                "  Types détectés :"
            )

            for pii_type, count in sorted(
                analysis["types"].items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            ):

                unique_count = (
                    analysis[
                        "unique_records_by_type"
                    ].get(
                        pii_type,
                        0,
                    )
                )

                print(
                    f"    {pii_type:<25} "
                    f"{count:<6} matchs | "
                    f"{unique_count} records"
                )

        print()

    if not analyses:

        raise RuntimeError(
            "Aucun dataset disponible."
        )

    report = build_json_report(
        analyses
    )

    json_path = (
        output_dir / "pii_report.json"
    )

    txt_path = (
        output_dir / "pii_report.txt"
    )

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_text_report(
        report,
        txt_path,
    )

    print("=" * 70)
    print("RAPPORT TERMINÉ")
    print("=" * 70)
    print()

    print(
        f"Records analysés       : "
        f"{report['summary']['total_records']}"
    )

    print(
        f"Records avec alertes   : "
        f"{report['summary']['records_with_alerts']}"
    )

    print(
        f"Matchs détectés        : "
        f"{report['summary']['total_matches']}"
    )

    print()
    print(
        f"JSON : {json_path}"
    )
    print(
        f"TXT  : {txt_path}"
    )
    print()


if __name__ == "__main__":
    main()
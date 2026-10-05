"""
anonymize_dpo.py

Anonymisation des datasets DPO / RewardBench avec Microsoft Presidio.

Entrées :
    data/processed/dpo_dataset_v_n/dpo_train.jsonl
    data/processed/dpo_dataset_v_n/dpo_validation.jsonl
    data/processed/medical_rewardbench.jsonl

Sorties :
    data/processed/dpo_dataset_v_n/anonymized/dpo_train.jsonl
    data/processed/dpo_dataset_v_n/anonymized/dpo_validation.jsonl
    data/processed/dpo_dataset_v_n/anonymized/medical_rewardbench.jsonl

Rapport :
    reports/dpo_dataset_v_n/presidio_anonymization_report.json
    reports/dpo_dataset_v_n/presidio_anonymization_report.txt

Important :
    - Les fichiers raw ne sont jamais modifiés.
    - Les IDs techniques et métadonnées de provenance sont conservés.
    - Les contenus prompt/chosen/rejected sont anonymisés.
    - Stratégie utilisée : replace.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT_DIR = Path("data/processed/dpo_dataset_v2")

DEFAULT_OUTPUT_DIR = (
    Path("data/processed/dpo_dataset_v2/anonymized")
)

DEFAULT_REPORT_DIR = Path("reports")


DATASETS = {
    "dpo_train": "dpo_train.jsonl",
    "dpo_validation": "dpo_validation.jsonl",
    "rewardbench": "medical_rewardbench.jsonl",
}


# ============================================================
# CHAMPS À PROTÉGER
# ============================================================

# Ces champs ne doivent pas être anonymisés.
#
# Ils servent à conserver l'intégrité technique du dataset.

PROTECTED_PATHS = {
    "id",
    "source",
    "provenance",
    "quality",

    "provenance.source_id",
    "provenance.processing_version",
    "provenance.dpo_split",

    "source.dataset",
    "source.split",
    "source.label_type",

    "quality.preference_valid",
    "quality.clinically_validated",
    "quality.clinical_validation_status",
}


# ============================================================
# CHAMPS À ANONYMISER
# ============================================================

# Les données réellement textuelles du DPO.

TARGET_ROOT_FIELDS = {
    "prompt",
    "chosen",
    "rejected",
}


# ============================================================
# ENTITÉS PRESIDIO
# ============================================================

# On se concentre sur les PII demandées / pertinentes.
#
# Les dates, codes postaux et URLs ne sont PAS anonymisés
# automatiquement ici afin d'éviter de dégrader les données
# médicales.

ENTITIES_TO_DETECT = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "LOCATION",
    "MEDICAL_LICENSE",
]


# ============================================================
# OPÉRATEURS
# ============================================================

# "replace" est explicitement demandé dans l'énoncé.
#
# Chaque type est remplacé par un token explicite.

OPERATORS = {
    "PERSON": OperatorConfig(
        "replace",
        {
            "new_value": "[PERSON]",
        },
    ),

    "EMAIL_ADDRESS": OperatorConfig(
        "replace",
        {
            "new_value": "[EMAIL]",
        },
    ),

    "PHONE_NUMBER": OperatorConfig(
        "replace",
        {
            "new_value": "[PHONE]",
        },
    ),

    "IP_ADDRESS": OperatorConfig(
        "replace",
        {
            "new_value": "[IP]",
        },
    ),

    "LOCATION": OperatorConfig(
        "replace",
        {
            "new_value": "[LOCATION]",
        },
    ),

    "MEDICAL_LICENSE": OperatorConfig(
        "replace",
        {
            "new_value": "[MEDICAL_LICENSE]",
        },
    ),
}


# ============================================================
# REGEX COMPLÉMENTAIRES
# ============================================================

# Presidio est utilisé en priorité.
#
# Ces regex servent uniquement de filet de sécurité pour les
# emails et téléphones qui pourraient échapper à Presidio.

EMAIL_FALLBACK = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)


PHONE_FALLBACK = re.compile(
    r"(?<!\w)"
    r"(?:\+|00)"
    r"\d{1,3}"
    r"(?:[\s().-]*\d{2,4}){2,6}"
    r"(?!\w)"
)


# ============================================================
# INITIALISATION PRESIDIO
# ============================================================

def create_analyzer() -> AnalyzerEngine:
    """
    Initialise Presidio avec les moteurs linguistiques
    français et anglais.
    """

    analyzer = AnalyzerEngine(
        default_score_threshold=0.35,
    )

    return analyzer


def create_anonymizer() -> AnonymizerEngine:
    """
    Initialise le moteur d'anonymisation.
    """

    return AnonymizerEngine()


# ============================================================
# TEST DU TEXTE
# ============================================================

def analyze_text(
    text: str,
    analyzer: AnalyzerEngine,
    language: str,
) -> list[RecognizerResult]:
    """
    Analyse un texte avec Presidio.
    """

    try:

        results = analyzer.analyze(
            text=text,
            language=language,
            entities=ENTITIES_TO_DETECT,
        )

        return results

    except Exception as exc:

        print(
            f"[WARNING] Analyse Presidio échouée "
            f"(langue={language}) : {exc}"
        )

        return []


# ============================================================
# DÉTECTION DE LANGUE SIMPLE
# ============================================================

def detect_language(text: str) -> str:
    """
    Détection simple FR/EN.

    On privilégie quelques marqueurs linguistiques pour éviter
    d'ajouter une dépendance supplémentaire.

    La langue sert uniquement à sélectionner le modèle Presidio.
    """

    text_lower = text.lower()

    french_markers = {
        " le ",
        " la ",
        " les ",
        " des ",
        " une ",
        " un ",
        " avec ",
        " pour ",
        " dans ",
        " est ",
        " sont ",
        " patient ",
        " symptômes ",
        " médecin ",
        " traitement ",
        " maladie ",
    }

    english_markers = {
        " the ",
        " a ",
        " an ",
        " with ",
        " for ",
        " in ",
        " is ",
        " are ",
        " patient ",
        " symptoms ",
        " doctor ",
        " treatment ",
        " disease ",
    }

    french_score = sum(
        1
        for marker in french_markers
        if marker in f" {text_lower} "
    )

    english_score = sum(
        1
        for marker in english_markers
        if marker in f" {text_lower} "
    )

    if french_score > english_score:
        return "fr"

    return "en"


# ============================================================
# ANONYMISATION PRESIDIO
# ============================================================

def anonymize_text(
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Analyse puis anonymise un texte.

    Retourne :
        texte anonymisé
        liste des entités détectées
    """

    if not text.strip():
        return text, []

    language = detect_language(text)

    results = analyze_text(
        text,
        analyzer,
        language,
    )

    detected = []

    for result in results:

        detected.append(
            {
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": round(
                    float(result.score),
                    4,
                ),
                "language": language,
            }
        )

    # --------------------------------------------------------
    # FALLBACK EMAIL
    # --------------------------------------------------------

    existing_email_spans = {
        (
            result.start,
            result.end,
        )
        for result in results
        if result.entity_type == "EMAIL_ADDRESS"
    }

    for match in EMAIL_FALLBACK.finditer(text):

        span = (
            match.start(),
            match.end(),
        )

        if span not in existing_email_spans:

            results.append(
                RecognizerResult(
                    entity_type="EMAIL_ADDRESS",
                    start=match.start(),
                    end=match.end(),
                    score=1.0,
                )
            )

            detected.append(
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0,
                    "language": language,
                    "detector": "fallback_regex",
                }
            )

    # --------------------------------------------------------
    # FALLBACK PHONE
    # --------------------------------------------------------

    existing_phone_spans = {
        (
            result.start,
            result.end,
        )
        for result in results
        if result.entity_type == "PHONE_NUMBER"
    }

    for match in PHONE_FALLBACK.finditer(text):

        span = (
            match.start(),
            match.end(),
        )

        if span not in existing_phone_spans:

            results.append(
                RecognizerResult(
                    entity_type="PHONE_NUMBER",
                    start=match.start(),
                    end=match.end(),
                    score=1.0,
                )
            )

            detected.append(
                {
                    "entity_type": "PHONE_NUMBER",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0,
                    "language": language,
                    "detector": "fallback_regex",
                }
            )

    if not results:
        return text, []

    # --------------------------------------------------------
    # ANONYMISATION
    # --------------------------------------------------------

    try:

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=OPERATORS,
        )

        return (
            anonymized.text,
            detected,
        )

    except Exception as exc:

        print(
            f"[WARNING] Anonymisation Presidio échouée : "
            f"{exc}"
        )

        return text, detected


# ============================================================
# PARCOURS JSON
# ============================================================

def should_protect(
    path: str,
) -> bool:

    if path in PROTECTED_PATHS:
        return True

    for protected in PROTECTED_PATHS:

        if path.startswith(
            protected + "."
        ):
            return True

    return False


def process_value(
    value: Any,
    path: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    counters: Counter,
) -> Any:
    """
    Parcours récursivement le JSON.

    Seuls les champs texte des données DPO sont anonymisés.
    """

    # --------------------------------------------------------
    # CHAMP PROTÉGÉ
    # --------------------------------------------------------

    if should_protect(path):

        return value

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if isinstance(value, str):

        anonymized_text, detected = anonymize_text(
            value,
            analyzer,
            anonymizer,
        )

        for entity in detected:

            counters[
                entity["entity_type"]
            ] += 1

        return anonymized_text

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    if isinstance(value, list):

        result = []

        for index, item in enumerate(value):

            child_path = (
                f"{path}[{index}]"
            )

            result.append(
                process_value(
                    item,
                    child_path,
                    analyzer,
                    anonymizer,
                    counters,
                )
            )

        return result

    # --------------------------------------------------------
    # DICT
    # --------------------------------------------------------

    if isinstance(value, dict):

        result = {}

        for key, item in value.items():

            if key == "_report_line_number":
                continue

            child_path = (
                f"{path}.{key}"
                if path
                else key
            )

            result[key] = process_value(
                item,
                child_path,
                analyzer,
                anonymizer,
                counters,
            )

        return result

    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    return value


# ============================================================
# JSONL
# ============================================================

def process_jsonl(
    input_path: Path,
    output_path: Path,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
) -> dict[str, Any]:
    """
    Traite un fichier JSONL complet.
    """

    total_records = 0
    records_anonymized = 0

    entity_counter = Counter()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as infile, output_path.open(
        "w",
        encoding="utf-8",
    ) as outfile:

        for line_number, line in enumerate(
            infile,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"JSON invalide : "
                    f"{input_path}:{line_number}"
                ) from exc

            if not isinstance(record, dict):

                raise ValueError(
                    f"Record invalide : "
                    f"{input_path}:{line_number}"
                )

            total_records += 1

            before_counter = sum(
                entity_counter.values()
            )

            anonymized_record = process_value(
                record,
                "",
                analyzer,
                anonymizer,
                entity_counter,
            )

            after_counter = sum(
                entity_counter.values()
            )

            if after_counter > before_counter:

                records_anonymized += 1

            outfile.write(
                json.dumps(
                    anonymized_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "total_records": total_records,
        "records_anonymized": records_anonymized,
        "entities": dict(entity_counter),
    }


# ============================================================
# CONTRÔLE POST-ANONYMISATION
# ============================================================

def scan_remaining_pii(
    path: Path,
    analyzer: AnalyzerEngine,
) -> dict[str, Any]:
    """
    Deuxième passage Presidio pour vérifier ce qui subsiste.
    """

    remaining_counter = Counter()
    remaining_records = 0
    total_matches = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as infile:

        for line_number, line in enumerate(
            infile,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            matches_for_record = []

            for field_path, text in walk_scannable_text(
                record
            ):

                language = detect_language(text)

                results = analyze_text(
                    text,
                    analyzer,
                    language,
                )

                for result in results:

                    remaining_counter[
                        result.entity_type
                    ] += 1

                    matches_for_record.append(
                        result.entity_type
                    )

            if matches_for_record:

                remaining_records += 1
                total_matches += len(
                    matches_for_record
                )

    return {
        "remaining_records": remaining_records,
        "remaining_matches": total_matches,
        "remaining_entities": dict(
            remaining_counter
        ),
    }


# ============================================================
# WALK TEXTE POST-SCAN
# ============================================================

def walk_scannable_text(
    value: Any,
    path: str = "",
):
    """
    Retourne les textes à scanner après anonymisation.
    """

    if isinstance(value, str):

        if not should_protect(path):

            yield path, value

        return

    if isinstance(value, dict):

        for key, child in value.items():

            child_path = (
                f"{path}.{key}"
                if path
                else key
            )

            if should_protect(child_path):
                continue

            yield from walk_scannable_text(
                child,
                child_path,
            )

        return

    if isinstance(value, list):

        for index, child in enumerate(value):

            child_path = (
                f"{path}[{index}]"
            )

            yield from walk_scannable_text(
                child,
                child_path,
            )


# ============================================================
# RAPPORT
# ============================================================

def write_report(
    results: list[dict[str, Any]],
    output_json: Path,
    output_txt: Path,
) -> None:

    total_records = sum(
        item["total_records"]
        for item in results
    )

    total_anonymized = sum(
        item["records_anonymized"]
        for item in results
    )

    total_entities = Counter()

    for item in results:

        for entity, count in item[
            "entities"
        ].items():

            total_entities[entity] += count

    report = {
        "report_version": "presidio_dpo_v1",

        "strategy": "replace",

        "protected_fields": sorted(
            PROTECTED_PATHS
        ),

        "summary": {
            "total_records": total_records,
            "records_anonymized": (
                total_anonymized
            ),
            "total_entities_detected": sum(
                total_entities.values()
            ),
        },

        "entities": dict(
            sorted(
                total_entities.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ),

        "datasets": results,
    }

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = []

    lines.append("=" * 80)
    lines.append(
        "RAPPORT D'ANONYMISATION PRESIDIO"
    )
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        f"Stratégie : {report['strategy']}"
    )

    lines.append("")

    lines.append("=== SYNTHÈSE ===")
    lines.append("")

    lines.append(
        f"Records traités       : "
        f"{total_records}"
    )

    lines.append(
        f"Records anonymisés    : "
        f"{total_anonymized}"
    )

    lines.append(
        f"Entités détectées     : "
        f"{sum(total_entities.values())}"
    )

    lines.append("")

    lines.append(
        "=== ENTITÉS DÉTECTÉES ==="
    )
    lines.append("")

    for entity, count in sorted(
        total_entities.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):

        lines.append(
            f"{entity:<30} {count}"
        )

    lines.append("")

    lines.append(
        "=== PAR DATASET ==="
    )
    lines.append("")

    for item in results:

        lines.append(
            f"[{Path(item['input']).name}]"
        )

        lines.append(
            f"  Records       : "
            f"{item['total_records']}"
        )

        lines.append(
            f"  Anonymisés    : "
            f"{item['records_anonymized']}"
        )

        for entity, count in sorted(
            item["entities"].items(),
            key=lambda x: (
                -x[1],
                x[0],
            ),
        ):

            lines.append(
                f"    - {entity:<25} "
                f"{count}"
            )

        lines.append("")

    lines.append(
        "=== CHAMPS PROTÉGÉS ==="
    )
    lines.append("")

    for field in sorted(
        PROTECTED_PATHS
    ):

        lines.append(
            f"- {field}"
        )

    lines.append("")

    lines.append(
        "Les fichiers sources n'ont pas été modifiés."
    )

    output_txt.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Anonymise les datasets DPO/RewardBench "
            "avec Microsoft Presidio."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    report_dir = args.report_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "ANONYMISATION PRESIDIO — DPO"
    )
    print("=" * 70)
    print()

    print(
        "Initialisation Presidio..."
    )

    analyzer = create_analyzer()

    anonymizer = create_anonymizer()

    print(
        "Presidio initialisé."
    )

    print()

    results = []

    for dataset_name, filename in DATASETS.items():

        input_path = (
            input_dir / filename
        )

        output_path = (
            output_dir / filename
        )

        print(
            f"Traitement : {dataset_name}"
        )

        print(
            f"  Input  : {input_path}"
        )

        print(
            f"  Output : {output_path}"
        )

        if not input_path.exists():

            print(
                f"  [WARNING] "
                f"Fichier absent."
            )

            continue

        result = process_jsonl(
            input_path,
            output_path,
            analyzer,
            anonymizer,
        )

        results.append(result)

        print(
            f"  Records       : "
            f"{result['total_records']}"
        )

        print(
            f"  Anonymisés    : "
            f"{result['records_anonymized']}"
        )

        print(
            f"  Entités       : "
            f"{sum(result['entities'].values())}"
        )

        for entity, count in sorted(
            result["entities"].items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            print(
                f"    {entity:<25} "
                f"{count}"
            )

        print()

    if not results:

        raise RuntimeError(
            "Aucun dataset n'a été traité."
        )

    # --------------------------------------------------------
    # RAPPORT
    # --------------------------------------------------------

    json_report = (
        report_dir
        / "presidio_anonymization_report.json"
    )

    txt_report = (
        report_dir
        / "presidio_anonymization_report.txt"
    )

    write_report(
        results,
        json_report,
        txt_report,
    )

    print("=" * 70)
    print(
        "ANONYMISATION TERMINÉE"
    )
    print("=" * 70)
    print()

    print(
        f"JSON report : {json_report}"
    )

    print(
        f"TXT report  : {txt_report}"
    )

    print()

    print(
        "Datasets anonymisés :"
    )

    for result in results:

        print(
            f"  {result['output']}"
        )

    print()

    print(
        "Les fichiers originaux n'ont pas été modifiés."
    )

    print()


if __name__ == "__main__":
    main()
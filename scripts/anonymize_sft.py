import argparse
import json
from pathlib import Path
from collections import Counter

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/sft_5000.jsonl")
OUTPUT_FILE = Path("data/processed/sft_5000_anonymized.jsonl")

SUPPORTED_LANGUAGES = {
    "fr": "fr_core_news_md",
    "en": "en_core_web_sm",
}


# Entités PII que nous voulons rechercher.
PII_ENTITIES = [
    #"PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    #"LOCATION",
    #"DATE_TIME",
    "URL",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
]


# ============================================================
# CHAMPS PROTÉGÉS
# ============================================================
#
# Ces champs sont des métadonnées techniques / de traçabilité.
# Presidio ne doit jamais les modifier.
#
# Important :
# language et task sont nécessaires à la stratification SFT.
# Ils doivent donc impérativement rester inchangés.
#
# Exemple :
#
# "language": "fr"      -> doit rester "fr"
# "task": "medical_mcq" -> doit rester "medical_mcq"
#
# ============================================================

PROTECTED_KEYS = {
    # Identifiants
    "id",
    "original_id",

    # Dataset / provenance
    "dataset",
    "source",
    "url",
    "source_license",
    "source_id",

    # Version / traitement
    "processing_version",

    # Métadonnées nécessaires aux splits
    "language",
    "lang",
    "task",
    "type",

    # Métadonnées de qualité
    "quality",

    # Statuts / contrôles
    "dpo_split",
    "split",
    "label_type",
    "preference_valid",
    "clinically_validated",
    "clinical_validation_status",
    "expert_reviewed_labels",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Détection et anonymisation PII avec Microsoft Presidio."
    )

    parser.add_argument(
        "--strategy",
        choices=["replace", "mask", "redact"],
        default="replace",
        help="Stratégie d'anonymisation."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Fichier JSONL d'entrée."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Fichier JSONL anonymisé."
    )

    return parser.parse_args()


# ============================================================
# PRESIDIO
# ============================================================

def create_analyzer():
    """
    Crée un AnalyzerEngine bilingue FR/EN avec spaCy.
    """

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "fr",
                "model_name": "fr_core_news_md",
            },
            {
                "lang_code": "en",
                "model_name": "en_core_web_sm",
            },
        ],
    }

    provider = NlpEngineProvider(
        nlp_configuration=configuration
    )

    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["fr", "en"],
    )

    return analyzer


def create_anonymizer():
    return AnonymizerEngine()


# ============================================================
# ANONYMISATION
# ============================================================

def build_operators(strategy):
    """
    Configure la stratégie demandée.

    replace :
        John Doe -> <PERSON>

    mask :
        John Doe -> ********

    redact :
        John Doe -> ""
    """

    if strategy == "replace":

        operators = {
            entity: OperatorConfig(
                "replace",
                {
                    "new_value": f"<{entity}>"
                },
            )
            for entity in PII_ENTITIES
        }

    elif strategy == "mask":

        operators = {
            entity: OperatorConfig(
                "mask",
                {
                    "masking_char": "*",
                    "chars_to_mask": 1000,
                    "from_end": False,
                },
            )
            for entity in PII_ENTITIES
        }

    elif strategy == "redact":

        operators = {
            entity: OperatorConfig(
                "redact",
                {},
            )
            for entity in PII_ENTITIES
        }

    else:
        raise ValueError(
            f"Stratégie inconnue : {strategy}"
        )

    return operators


def anonymize_text(
    text,
    language,
    analyzer,
    anonymizer,
    operators,
    stats,
):
    """
    Analyse puis anonymise un texte.
    """

    if not isinstance(text, str) or not text.strip():
        return text

    if language not in SUPPORTED_LANGUAGES:
        stats["unsupported_language"] += 1
        return text

    results = analyzer.analyze(
        text=text,
        language=language,
        entities=PII_ENTITIES,
    )

    if not results:
        return text

    for result in results:
        stats["detected_entities"][result.entity_type] += 1

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )

    stats["texts_anonymized"] += 1

    return anonymized.text


# ============================================================
# TRAITEMENT JSON RÉCURSIF
# ============================================================

def process_value(
    value,
    key,
    language,
    analyzer,
    anonymizer,
    operators,
    stats,
):
    """
    Parcourt récursivement le JSON.

    Les métadonnées protégées ne sont pas modifiées.
    """

    # --------------------------------------------------------
    # Champ protégé
    # --------------------------------------------------------

    if key in PROTECTED_KEYS:
        return value

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(value, str):

        return anonymize_text(
            text=value,
            language=language,
            analyzer=analyzer,
            anonymizer=anonymizer,
            operators=operators,
            stats=stats,
        )

    # --------------------------------------------------------
    # Liste
    # --------------------------------------------------------

    if isinstance(value, list):

        return [
            process_value(
                item,
                key=None,
                language=language,
                analyzer=analyzer,
                anonymizer=anonymizer,
                operators=operators,
                stats=stats,
            )
            for item in value
        ]

    # --------------------------------------------------------
    # Dictionnaire
    # --------------------------------------------------------

    if isinstance(value, dict):

        result = {}

        for child_key, child_value in value.items():

            result[child_key] = process_value(
                child_value,
                key=child_key,
                language=language,
                analyzer=analyzer,
                anonymizer=anonymizer,
                operators=operators,
                stats=stats,
            )

        return result

    # --------------------------------------------------------
    # Autres types
    # --------------------------------------------------------

    return value


# ============================================================
# CONTRÔLE PII APRÈS ANONYMISATION
# ============================================================

def check_remaining_pii(
    value,
    key,
    language,
    analyzer,
    stats,
):
    """
    Deuxième passage Presidio.

    Objectif :
        vérifier qu'aucune PII détectable ne reste.
    """

    # --------------------------------------------------------
    # Champ protégé
    # --------------------------------------------------------

    if key in PROTECTED_KEYS:
        return

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(value, str):

        if not value.strip():
            return

        if language not in SUPPORTED_LANGUAGES:
            return

        results = analyzer.analyze(
            text=value,
            language=language,
            entities=PII_ENTITIES,
        )

        for result in results:
            stats["remaining_pii"][result.entity_type] += 1

        return

    # --------------------------------------------------------
    # Liste
    # --------------------------------------------------------

    if isinstance(value, list):

        for item in value:

            check_remaining_pii(
                item,
                key=None,
                language=language,
                analyzer=analyzer,
                stats=stats,
            )

        return

    # --------------------------------------------------------
    # Dictionnaire
    # --------------------------------------------------------

    if isinstance(value, dict):

        for child_key, child_value in value.items():

            check_remaining_pii(
                child_value,
                key=child_key,
                language=language,
                analyzer=analyzer,
                stats=stats,
            )


# ============================================================
# CONTRÔLE DES MÉTADONNÉES
# ============================================================

def check_protected_metadata(original_rows, anonymized_rows):
    """
    Vérifie que les métadonnées importantes n'ont pas été
    modifiées par l'anonymisation.
    """

    errors = []

    if len(original_rows) != len(anonymized_rows):
        errors.append(
            "Le nombre de lignes a changé."
        )
        return errors

    protected_fields = [
        "id",
        "language",
        "task",
        "source",
        "provenance",
        "quality",
    ]

    for index, (original, anonymized) in enumerate(
        zip(original_rows, anonymized_rows)
    ):

        for field in protected_fields:

            original_value = original.get(field)
            anonymized_value = anonymized.get(field)

            if original_value != anonymized_value:

                errors.append(
                    f"Ligne {index + 1}: "
                    f"champ protégé '{field}' modifié."
                )

    return errors


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("=" * 70)
    print("ANONYMISATION PII - MICROSOFT PRESIDIO")
    print("=" * 70)

    print(f"Entrée     : {args.input}")
    print(f"Sortie     : {args.output}")
    print(f"Stratégie  : {args.strategy}")
    print()

    if not args.input.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {args.input}"
        )

    # --------------------------------------------------------
    # Création des moteurs
    # --------------------------------------------------------

    print("[1] Initialisation Presidio...")

    analyzer = create_analyzer()
    anonymizer = create_anonymizer()
    operators = build_operators(args.strategy)

    print("OK : AnalyzerEngine")
    print("OK : AnonymizerEngine")
    print("OK : modèle français fr_core_news_md")
    print("OK : modèle anglais en_core_web_sm")
    print()

    # --------------------------------------------------------
    # Lecture
    # --------------------------------------------------------

    print("[2] Lecture du JSONL...")

    rows = []

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)
                rows.append(row)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"JSON invalide ligne {line_number}: {exc}"
                ) from exc

    print(f"Entrées chargées : {len(rows)}")
    print()

    # --------------------------------------------------------
    # Vérification nombre attendu
    # --------------------------------------------------------

    if len(rows) != 5000:

        raise ValueError(
            f"Nombre inattendu d'entrées : {len(rows)} "
            f"(attendu : 5000)"
        )

    # --------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------

    stats = {
        "detected_entities": Counter(),
        "remaining_pii": Counter(),
        "texts_anonymized": 0,
        "unsupported_language": 0,
    }

    # --------------------------------------------------------
    # Anonymisation
    # --------------------------------------------------------

    print("[3] Anonymisation...")

    anonymized_rows = []

    for row in rows:

        language = row.get(
            "language",
            ""
        )

        if isinstance(language, str):
            language = language.lower().strip()
        else:
            language = ""

        anonymized_row = process_value(
            value=row,
            key=None,
            language=language,
            analyzer=analyzer,
            anonymizer=anonymizer,
            operators=operators,
            stats=stats,
        )

        anonymized_rows.append(
            anonymized_row
        )

    print("Anonymisation terminée.")
    print()

    # --------------------------------------------------------
    # Vérification métadonnées
    # --------------------------------------------------------

    print("[4] Vérification des métadonnées protégées...")
    print("-" * 70)

    metadata_errors = check_protected_metadata(
        rows,
        anonymized_rows,
    )

    if metadata_errors:

        for error in metadata_errors[:20]:
            print(f"❌ {error}")

        if len(metadata_errors) > 20:
            print(
                f"... et {len(metadata_errors) - 20} autre(s) erreur(s)."
            )

        raise ValueError(
            "Les métadonnées protégées ont été modifiées."
        )

    print("✅ Métadonnées protégées inchangées.")
    print("✅ language conservé.")
    print("✅ task conservé.")
    print("✅ source conservé.")
    print("✅ provenance conservée.")
    print("✅ quality conservé.")
    print()

    # --------------------------------------------------------
    # Création du dossier
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Écriture
    # --------------------------------------------------------

    print("[5] Écriture du fichier anonymisé...")

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in anonymized_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Fichier créé : {args.output}")
    print()

    # --------------------------------------------------------
    # Second contrôle PII
    # --------------------------------------------------------

    print("[6] CONTRÔLE PII APRÈS ANONYMISATION")
    print("-" * 70)

    for row in anonymized_rows:

        language = row.get(
            "language",
            ""
        )

        if isinstance(language, str):
            language = language.lower().strip()
        else:
            language = ""

        check_remaining_pii(
            value=row,
            key=None,
            language=language,
            analyzer=analyzer,
            stats=stats,
        )

    # --------------------------------------------------------
    # Rapport
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RAPPORT PRESIDIO")
    print("=" * 70)

    print()
    print("PII détectées AVANT anonymisation :")

    if stats["detected_entities"]:

        for entity, count in sorted(
            stats["detected_entities"].items()
        ):

            print(
                f"  {entity:<20}: {count}"
            )

    else:

        print("  Aucune")

    print()
    print(
        f"Textes anonymisés : "
        f"{stats['texts_anonymized']}"
    )

    print()
    print("PII détectées APRÈS anonymisation :")

    if stats["remaining_pii"]:

        for entity, count in sorted(
            stats["remaining_pii"].items()
        ):

            print(
                f"  {entity:<20}: {count}"
            )

    else:

        print("  Aucune")

    print()

    if stats["unsupported_language"]:

        print(
            "⚠️ Textes avec langue non supportée : "
            f"{stats['unsupported_language']}"
        )

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    remaining_total = sum(
        stats["remaining_pii"].values()
    )

    print()
    print("=" * 70)

    if remaining_total == 0:

        print("✅ CONTROLE PII : OK")

        print(
            "Aucune PII détectable par Presidio "
            "ne subsiste dans les champs analysés."
        )

    else:

        print("⚠️ CONTROLE PII : À VÉRIFIER")

        print(
            f"{remaining_total} occurrence(s) "
            "PII restent détectées."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
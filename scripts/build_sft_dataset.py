from pathlib import Path
import json
import hashlib
import random
import xml.etree.ElementTree as ET


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ============================================================
# OUTILS
# ============================================================

def clean_text(value):
    """Nettoie un texte sans inventer de contenu."""
    if value is None:
        return ""

    if isinstance(value, str):
        return " ".join(value.split())

    return str(value).strip()


def make_id(source, original_id):
    """Crée un identifiant stable."""
    raw = f"{source}:{original_id}"

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def ensure_list(value):
    """Transforme une valeur en liste."""
    if value is None:
        return []

    if isinstance(value, list):
        return [
            clean_text(x)
            for x in value
            if clean_text(x)
        ]

    value = clean_text(value)

    if not value:
        return []

    if "," in value:
        return [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    return [value]


def deduplicate(records):
    """Supprime les doublons sur instruction + response."""
    seen = set()
    result = []

    for record in records:

        key = (
            record.get(
                "instruction",
                ""
            ).strip().lower(),

            record.get(
                "response",
                ""
            ).strip().lower()
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(record)

    return result


def sample_records(records, n, seed=SEED):
    """Sélection déterministe de n éléments."""

    if len(records) < n:
        raise ValueError(
            f"Pas assez de données : "
            f"{len(records)} disponibles, "
            f"{n} demandées."
        )

    rng = random.Random(seed)

    return rng.sample(records, n)


def save_jsonl(records, path):
    """Sauvegarde en JSONL UTF-8."""

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# CONSTRUCTION QCM
# ============================================================

def build_mcq_instruction(question, choices):
    """
    Construit l'entrée visible par le modèle pour un QCM.

    Le modèle voit :
        Question
        + choix A, B, C, D, E

    Les choix ne sont jamais inventés.
    """

    question = clean_text(question)

    lines = [
        question,
        "",
        "Options:"
    ]

    for letter in [
        "A",
        "B",
        "C",
        "D",
        "E"
    ]:

        choice = clean_text(
            choices.get(letter)
        )

        if choice:
            lines.append(
                f"{letter}. {choice}"
            )

    return "\n".join(lines)


def build_mcq_response(correct_answers, choices):
    """
    Construit la réponse cible du QCM.

    Exemple :
        Réponse correcte :
        A. Texte de la proposition A

    Pour plusieurs bonnes réponses :
        Réponses correctes :
        A. ...
        C. ...

    Le texte provient uniquement des choix
    présents dans la source.
    """

    if not correct_answers:

        return (
            "Réponse correcte "
            "non renseignée."
        )

    answer_lines = []

    for answer in correct_answers:

        letter = clean_text(
            answer
        ).upper()

        choice = clean_text(
            choices.get(letter)
        )

        if choice:

            answer_lines.append(
                f"{letter}. {choice}"
            )

        else:

            # On conserve la réponse originale
            # même si le choix correspondant
            # n'est pas disponible.
            answer_lines.append(
                letter
            )

    if len(answer_lines) == 1:

        return (
            "Réponse correcte : "
            + answer_lines[0]
        )

    return (
        "Réponses correctes :\n"
        + "\n".join(answer_lines)
    )


# ============================================================
# MEDQUAD
# ============================================================

def load_medquad():

    print("\n=== MEDQUAD ===")

    medquad_dir = RAW_DIR / "MedQuAD"

    if not medquad_dir.exists():
        raise FileNotFoundError(
            f"Dossier MedQuAD introuvable : "
            f"{medquad_dir}"
        )

    xml_files = list(
        medquad_dir.rglob("*.xml")
    )

    print(
        f"Fichiers XML trouvés : "
        f"{len(xml_files)}"
    )

    records = []
    answers_count = 0

    for xml_file in xml_files:

        try:

            tree = ET.parse(xml_file)
            root = tree.getroot()

        except Exception as e:

            print(
                f"[WARN] XML illisible : "
                f"{xml_file} -> {e}"
            )

            continue

        focus = clean_text(
            root.findtext("Focus")
        )

        # ----------------------------------------------------
        # UMLS
        # ----------------------------------------------------

        umls_cuis = []

        for cui in root.findall(
            "./FocusAnnotations/UMLS/CUIs/CUI"
        ):

            value = clean_text(
                cui.text
            )

            if value:
                umls_cuis.append(value)

        semantic_types = []

        for semantic_type in root.findall(
            "./FocusAnnotations/UMLS/"
            "SemanticTypes/SemanticType"
        ):

            value = clean_text(
                semantic_type.text
            )

            if value:
                semantic_types.append(value)

        semantic_group = clean_text(
            root.findtext(
                "./FocusAnnotations/UMLS/"
                "SemanticGroup"
            )
        )

        # ----------------------------------------------------
        # Q/A
        # ----------------------------------------------------

        qa_pairs = root.findall(
            "./QAPairs/QAPair"
        )

        for qa in qa_pairs:

            question_node = qa.find(
                "Question"
            )

            answer_node = qa.find(
                "Answer"
            )

            if question_node is None:
                continue

            question = clean_text(
                "".join(
                    question_node.itertext()
                )
            )

            if answer_node is None:
                continue

            answer = clean_text(
                "".join(
                    answer_node.itertext()
                )
            )

            if not question or not answer:
                continue

            answers_count += 1

            qid = question_node.attrib.get(
                "qid",
                qa.attrib.get("pid", "")
            )

            qtype = question_node.attrib.get(
                "qtype",
                ""
            )

            source = root.attrib.get(
                "source",
                ""
            )

            url = root.attrib.get(
                "url",
                ""
            )

            record = {

                "id": make_id(
                    "medquad",
                    f"{xml_file.relative_to(medquad_dir).as_posix()}:{qid}"
                ),

                "instruction": question,

                "response": answer,

                "language": "en",

                "task": "medical_qa",

                "source": {
                    "dataset": "MedQuAD",
                    "original_id": qid,
                    "source": source,
                    "url": url
                },

                "clinical_context": {
                    "clinical_case": None,
                    "symptoms": [],
                    "history": [],
                    "vitals": [],
                    "diagnosis": None,
                    "medications": [],
                    "allergies": []
                },

                "source_specific": {

                    "question_type": (
                        qtype
                        if qtype
                        else None
                    ),

                    "question_focus": (
                        focus
                        if focus
                        else None
                    ),

                    "synonyms": [],

                    "umls_cui": umls_cuis,

                    "semantic_types": (
                        semantic_types
                    ),

                    "semantic_group": (
                        semantic_group
                        if semantic_group
                        else None
                    ),

                    "choices": {},

                    "correct_answers": []
                },

                "quality": {
                    "confidence": None,
                    "validated": False
                },

                "provenance": {
                    "source_license": None,
                    "processing_version": "v0.1"
                }
            }

            records.append(record)

    print(
        f"Entrées avec réponse : "
        f"{answers_count}"
    )

    records = deduplicate(records)

    print(
        f"Après dédoublonnage : "
        f"{len(records)}"
    )

    selected = sample_records(
        records,
        1667,
        seed=SEED
    )

    print(
        f"Retenues pour SFT : "
        f"{len(selected)}"
    )

    return selected


# ============================================================
# FRENCHMEDMCQA
# ============================================================

def load_frenchmedmcqa():

    print("\n=== FRENCHMEDMCQA ===")

    dataset_dir = (
        RAW_DIR / "frenchmedmcqa"
    )

    files = [
        "train.json",
        "dev.json",
        "test.json"
    ]

    records = []

    for filename in files:

        path = dataset_dir / filename

        if not path.exists():

            print(
                f"[WARN] Fichier absent : "
                f"{path}"
            )

            continue

        try:

            with path.open(
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception as e:

            raise RuntimeError(
                f"Impossible de lire "
                f"{path}: {e}"
            )

        if not isinstance(data, list):

            raise ValueError(
                f"Structure inattendue "
                f"dans {path}."
            )

        valid_count = 0

        for item in data:

            if not isinstance(item, dict):
                continue

            question = clean_text(
                item.get("question")
            )

            if not question:
                continue

            # ------------------------------------------------
            # CHOIX
            # ------------------------------------------------

            raw_answers = item.get(
                "answers",
                {}
            )

            choices = {}

            if isinstance(
                raw_answers,
                dict
            ):

                for letter in [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e"
                ]:

                    value = clean_text(
                        raw_answers.get(letter)
                    )

                    if value:
                        choices[
                            letter.upper()
                        ] = value

            # ------------------------------------------------
            # RÉPONSES CORRECTES
            # ------------------------------------------------

            correct_answers = ensure_list(
                item.get(
                    "correct_answers"
                )
            )

            correct_answers = [
                answer.upper()
                for answer in correct_answers
            ]

            # ------------------------------------------------
            # CONSTRUCTION QCM
            # ------------------------------------------------

            instruction = build_mcq_instruction(
                question,
                choices
            )

            response = build_mcq_response(
                correct_answers,
                choices
            )

            subject = clean_text(
                item.get(
                    "subject_name"
                )
            )

            question_type = clean_text(
                item.get(
                    "type"
                )
            )

            original_id = (
                item.get("id")
                or f"{filename}:{valid_count}"
            )

            record = {

                "id": make_id(
                    "frenchmedmcqa",
                    original_id
                ),

                "instruction": instruction,

                "response": response,

                "language": "fr",

                "task": "medical_mcq",

                "source": {
                    "dataset": (
                        "FrenchMedMCQA"
                    ),

                    "original_id": str(
                        original_id
                    ),

                    "split": filename
                },

                "clinical_context": {
                    "clinical_case": None,
                    "symptoms": [],
                    "history": [],
                    "vitals": [],
                    "diagnosis": None,
                    "medications": [],
                    "allergies": []
                },

                "source_specific": {

                    "question_type": (
                        question_type
                        if question_type
                        else None
                    ),

                    "question_focus": None,

                    "synonyms": [],

                    "umls_cui": None,

                    "choices": choices,

                    "correct_answers": (
                        correct_answers
                    ),

                    "medical_subject": (
                        subject
                        if subject
                        else None
                    )
                },

                "quality": {
                    "confidence": None,
                    "validated": False
                },

                "provenance": {
                    "source_license": None,
                    "processing_version": "v0.1"
                }
            }

            records.append(record)

            valid_count += 1

        print(
            f"{filename} : "
            f"{valid_count} entrées valides"
        )

    print(
        f"Total chargé : "
        f"{len(records)}"
    )

    records = deduplicate(records)

    print(
        f"Après dédoublonnage : "
        f"{len(records)}"
    )

    selected = sample_records(
        records,
        1667,
        seed=SEED
    )

    print(
        f"Retenues pour SFT : "
        f"{len(selected)}"
    )

    return selected


# ============================================================
# MEDIQAL
# ============================================================

def load_mediqal():

    print("\n=== MEDIQAL ===")

    dataset_dir = (
        RAW_DIR / "mediqa"
    )

    files = [
        "train.json",
        "validation.json",
        "test.json"
    ]

    records = []

    for filename in files:

        path = dataset_dir / filename

        if not path.exists():

            print(
                f"[WARN] Fichier absent : "
                f"{path}"
            )

            continue

        valid_count = 0
        mcq_count = 0

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    item = json.loads(
                        line
                    )

                except json.JSONDecodeError as e:

                    print(
                        f"[WARN] Ligne JSON "
                        f"invalide dans "
                        f"{filename}: {e}"
                    )

                    continue

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                task = clean_text(
                    item.get("task")
                ).upper()

                # QCU / QCM / MCQU
                if task not in {
                    "QCU",
                    "QCM",
                    "MCQU"
                }:
                    continue

                mcq_count += 1

                question = clean_text(
                    item.get("question")
                )

                if not question:
                    continue

                # ------------------------------------------------
                # CHOIX
                # ------------------------------------------------

                choices = {}

                for letter in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E"
                ]:

                    key = (
                        f"answer_"
                        f"{letter.lower()}"
                    )

                    value = clean_text(
                        item.get(key)
                    )

                    if value:
                        choices[
                            letter
                        ] = value

                # ------------------------------------------------
                # RÉPONSES CORRECTES
                # ------------------------------------------------

                correct_answers = ensure_list(
                    item.get(
                        "correct_answers"
                    )
                )

                correct_answers = [
                    answer.upper()
                    for answer in correct_answers
                ]

                # ------------------------------------------------
                # CONSTRUCTION QCM
                # ------------------------------------------------

                instruction = build_mcq_instruction(
                    question,
                    choices
                )

                response = build_mcq_response(
                    correct_answers,
                    choices
                )

                clinical_case = clean_text(
                    item.get(
                        "clinical_case"
                    )
                )

                medical_subject = clean_text(
                    item.get(
                        "medical_subject"
                    )
                )

                question_type = clean_text(
                    item.get(
                        "question_type"
                    )
                )

                original_id = (
                    item.get("id")
                    or f"{filename}:{valid_count}"
                )

                record = {

                    "id": make_id(
                        "mediqal",
                        original_id
                    ),

                    "instruction": instruction,

                    "response": response,

                    "language": "fr",

                    "task": "medical_mcq",

                    "source": {
                        "dataset": "MediQAl",
                        "original_id": str(
                            original_id
                        ),
                        "split": filename
                    },

                    "clinical_context": {

                        "clinical_case": (
                            clinical_case
                            if clinical_case
                            else None
                        ),

                        "symptoms": [],
                        "history": [],
                        "vitals": [],
                        "diagnosis": None,
                        "medications": [],
                        "allergies": []
                    },

                    "source_specific": {

                        "question_type": (
                            question_type
                            if question_type
                            else None
                        ),

                        "question_focus": None,

                        "synonyms": [],

                        "umls_cui": None,

                        "choices": choices,

                        "correct_answers": (
                            correct_answers
                        ),

                        "medical_subject": (
                            medical_subject
                            if medical_subject
                            else None
                        )
                    },

                    "quality": {
                        "confidence": None,
                        "validated": False
                    },

                    "provenance": {
                        "source_license": None,
                        "processing_version": "v0.1"
                    }
                }

                records.append(record)

                valid_count += 1

        print(
            f"{filename} : "
            f"{valid_count} entrées MCQ"
        )

    print(
        f"Total MCQ chargé : "
        f"{len(records)}"
    )

    records = deduplicate(records)

    print(
        f"Après dédoublonnage : "
        f"{len(records)}"
    )

    selected = sample_records(
        records,
        1666,
        seed=SEED
    )

    print(
        f"Retenues pour SFT : "
        f"{len(selected)}"
    )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 46)

    print(
        "Construction du dataset "
        "SFT médical - 5 000"
    )

    print("=" * 46)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    medquad = load_medquad()

    frenchmedmcqa = (
        load_frenchmedmcqa()
    )

    mediqal = load_mediqal()

    # --------------------------------------------------------
    # Assemblage
    # --------------------------------------------------------

    all_records = (
        medquad
        + frenchmedmcqa
        + mediqal
    )

    print("\n=== ASSEMBLAGE ===")

    print(
        f"MedQuAD        : "
        f"{len(medquad)}"
    )

    print(
        f"FrenchMedMCQA  : "
        f"{len(frenchmedmcqa)}"
    )

    print(
        f"MediQAl        : "
        f"{len(mediqal)}"
    )

    print(
        f"TOTAL          : "
        f"{len(all_records)}"
    )

    if len(all_records) != 5000:

        raise ValueError(
            f"Le dataset final contient "
            f"{len(all_records)} entrées "
            f"au lieu de 5000."
        )

    # --------------------------------------------------------
    # Dédoublonnage global
    # --------------------------------------------------------

    final_records = deduplicate(
        all_records
    )

    if len(final_records) != 5000:

        raise ValueError(
            "Le dédoublonnage global a supprimé "
            f"{5000 - len(final_records)} "
            "entrées. Le dataset final "
            "n'est donc plus composé de "
            "5000 entrées."
        )

    # --------------------------------------------------------
    # Fichiers
    # --------------------------------------------------------

    medquad_path = (
        PROCESSED_DIR
        / "sft_medquad_1667.jsonl"
    )

    french_path = (
        PROCESSED_DIR
        / "sft_frenchmedmcqa_1667.jsonl"
    )

    mediqal_path = (
        PROCESSED_DIR
        / "sft_mediqal_1666.jsonl"
    )

    final_path = (
        PROCESSED_DIR
        / "sft_5000.jsonl"
    )

    save_jsonl(
        medquad,
        medquad_path
    )

    save_jsonl(
        frenchmedmcqa,
        french_path
    )

    save_jsonl(
        mediqal,
        mediqal_path
    )

    save_jsonl(
        final_records,
        final_path
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print(
        "\n=== FICHIERS CRÉÉS ==="
    )

    print(medquad_path)
    print(french_path)
    print(mediqal_path)
    print(final_path)

    print("\n" + "=" * 46)

    print(
        "Dataset SFT terminé : "
        "5 000 entrées"
    )

    print("=" * 46)


if __name__ == "__main__":
    main()
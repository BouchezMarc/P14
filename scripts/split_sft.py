import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("data/processed/sft_5000_anonymized.jsonl")

TRAIN_FILE = Path("data/processed/sft_train.jsonl")
VALIDATION_FILE = Path("data/processed/sft_validation.jsonl")
TEST_FILE = Path("data/processed/sft_test.jsonl")

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 3407

REQUIRED_FIELDS = [
    "id",
    "instruction",
    "response",
    "language",
    "task",
    "source",
]


# ============================================================================
# UTILITAIRES
# ============================================================================

def normalize(value):
    """Normalise une valeur pour les contrôles et les clés de stratification."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip().lower()

    return str(value).strip().lower()


def get_source_dataset(record):
    """
    Retourne le nom du dataset source.

    Le champ source peut être :
        {"dataset": "FrenchMedMCQA", ...}

    ou exceptionnellement une chaîne.
    """
    source = record.get("source")

    if isinstance(source, dict):
        return normalize(source.get("dataset"))

    return normalize(source)


def stratification_key(record):
    """
    Clé de stratification :

        source + language + task
    """
    source = get_source_dataset(record)
    language = normalize(record.get("language"))
    task = normalize(record.get("task"))

    return source, language, task


def record_hash(record):
    """
    Hash complet et déterministe du record.
    """
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def qa_hash(record):
    """
    Hash instruction + réponse.

    Permet de détecter une fuite de contenu entre splits,
    même si les IDs sont différents.
    """
    payload = (
        normalize(record.get("instruction"))
        + "\n"
        + normalize(record.get("response"))
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# CHARGEMENT
# ============================================================================

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide ligne {line_number} dans {path}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record non objet ligne {line_number} dans {path}"
                )

            records.append(record)

    return records


# ============================================================================
# CONTROLE DATASET SOURCE
# ============================================================================

def validate_source_dataset(records):
    print()
    print("=== CONTRÔLE DU DATASET SOURCE ===")

    print(f"Nombre total : {len(records)}")

    # ------------------------------------------------------------------------
    # Champs obligatoires
    # ------------------------------------------------------------------------

    missing = []

    for index, record in enumerate(records):
        for field in REQUIRED_FIELDS:
            if field not in record:
                missing.append((index, field))

    if missing:
        print("ERREUR : champs obligatoires manquants :")

        for index, field in missing[:20]:
            print(f"  record {index}: {field}")

        raise RuntimeError(
            f"{len(missing)} champ(s) obligatoire(s) manquant(s)."
        )

    print("Champs obligatoires : OK")

    # ------------------------------------------------------------------------
    # Clés de stratification
    # ------------------------------------------------------------------------

    invalid_strata = []

    for index, record in enumerate(records):
        key = stratification_key(record)

        if not all(key):
            invalid_strata.append((index, key))

    if invalid_strata:
        print("ERREUR : clés de stratification invalides :")

        for index, key in invalid_strata[:20]:
            print(f"  record {index}: {key}")

        raise RuntimeError(
            f"{len(invalid_strata)} record(s) ont une clé de stratification invalide."
        )

    print("Clés de stratification : OK")

    # ------------------------------------------------------------------------
    # Doublons exacts
    # ------------------------------------------------------------------------

    hashes = [record_hash(record) for record in records]

    duplicate_count = len(hashes) - len(set(hashes))

    print(f"Doublons exacts : {duplicate_count}")

    if duplicate_count:
        raise RuntimeError(
            f"{duplicate_count} doublon(s) exact(s) détecté(s)."
        )


# ============================================================================
# QUOTAS
# ============================================================================

def compute_stratified_quotas(groups, total_size):
    """
    Calcule les quotas train/validation/test en respectant simultanément :

        - les tailles exactes des strates ;
        - les tailles globales des splits ;
        - les ratios 80/10/10 au plus près.

    Méthode :

    1. Allocation du train par plus grands restes.
    2. Allocation de la validation sur les records restants
       par plus grands restes.
    3. Le test reçoit exactement le reliquat de chaque strate.

    Ainsi, pour chaque strate :

        train + validation + test == taille de la strate

    et globalement :

        train == 80 %
        validation == 10 %
        test == 10 %
    """

    # ------------------------------------------------------------------------
    # Cibles globales
    # ------------------------------------------------------------------------

    target_train = round(total_size * TRAIN_RATIO)
    target_validation = round(total_size * VALIDATION_RATIO)
    target_test = total_size - target_train - target_validation

    targets = {
        "train": target_train,
        "validation": target_validation,
        "test": target_test,
    }

    quotas = {
        key: {
            "train": 0,
            "validation": 0,
            "test": 0,
        }
        for key in sorted(groups)
    }

    # ------------------------------------------------------------------------
    # 1. Allocation du TRAIN
    # ------------------------------------------------------------------------

    train_remainders = []

    train_total_floor = 0

    for key in sorted(groups):
        size = len(groups[key])

        raw_train = size * TRAIN_RATIO
        base_train = int(raw_train)

        quotas[key]["train"] = base_train

        train_total_floor += base_train

        train_remainders.append(
            (
                raw_train - base_train,
                key,
            )
        )

    train_remaining = target_train - train_total_floor

    if train_remaining < 0:
        raise RuntimeError(
            "Le quota train calculé dépasse la cible globale."
        )

    train_remainders.sort(
        key=lambda item: (-item[0], item[1])
    )

    if train_remaining > len(train_remainders):
        raise RuntimeError(
            "Impossible d'allouer les restes du train."
        )

    for _, key in train_remainders[:train_remaining]:
        quotas[key]["train"] += 1

    # ------------------------------------------------------------------------
    # 2. Allocation de la VALIDATION
    #
    # On travaille maintenant sur les records restant dans chaque strate.
    # ------------------------------------------------------------------------

    validation_remainders = []

    validation_total_floor = 0

    for key in sorted(groups):
        size = len(groups[key])

        remaining_after_train = (
            size - quotas[key]["train"]
        )

        raw_validation = (
            size * VALIDATION_RATIO
        )

        # On conserve la cible proportionnelle originale.
        # Mais on ne peut évidemment pas dépasser les records
        # restant dans la strate.
        base_validation = min(
            int(raw_validation),
            remaining_after_train,
        )

        quotas[key]["validation"] = base_validation

        validation_total_floor += base_validation

        validation_remainders.append(
            (
                raw_validation - base_validation,
                key,
            )
        )

    validation_remaining = (
        target_validation - validation_total_floor
    )

    if validation_remaining < 0:
        raise RuntimeError(
            "Le quota validation calculé dépasse la cible globale."
        )

    # Une strate ne peut recevoir une unité supplémentaire en validation
    # que s'il lui reste encore au moins un record après train + validation.
    candidates = []

    for remainder, key in validation_remainders:
        size = len(groups[key])

        available = (
            size
            - quotas[key]["train"]
            - quotas[key]["validation"]
        )

        if available > 0:
            candidates.append(
                (
                    remainder,
                    key,
                )
            )

    candidates.sort(
        key=lambda item: (-item[0], item[1])
    )

    if validation_remaining > len(candidates):
        raise RuntimeError(
            "Impossible d'allouer les restes de la validation."
        )

    for _, key in candidates[:validation_remaining]:
        quotas[key]["validation"] += 1

    # ------------------------------------------------------------------------
    # 3. TEST = reliquat exact de chaque strate
    # ------------------------------------------------------------------------

    for key in sorted(groups):
        size = len(groups[key])

        quotas[key]["test"] = (
            size
            - quotas[key]["train"]
            - quotas[key]["validation"]
        )

        if quotas[key]["test"] < 0:
            raise RuntimeError(
                f"Quota test négatif pour la strate {key}."
            )

    # ------------------------------------------------------------------------
    # Vérification des lignes : chaque strate doit être conservée exactement
    # ------------------------------------------------------------------------

    for key in sorted(groups):
        original_size = len(groups[key])

        quota_total = (
            quotas[key]["train"]
            + quotas[key]["validation"]
            + quotas[key]["test"]
        )

        if quota_total != original_size:
            raise RuntimeError(
                f"Conservation invalide pour la strate {key}: "
                f"{quota_total} au lieu de {original_size}"
            )

    # ------------------------------------------------------------------------
    # Vérification des colonnes : tailles globales exactes
    # ------------------------------------------------------------------------

    final_totals = {
        split: sum(
            quotas[key][split]
            for key in quotas
        )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    if final_totals != targets:
        raise RuntimeError(
            "Les quotas finaux ne correspondent pas aux tailles cibles.\n"
            f"Attendu : {targets}\n"
            f"Obtenu  : {final_totals}"
        )

    return quotas, targets


# ============================================================================
# SPLIT STRATIFIE
# ============================================================================

def stratified_split(records, seed):
    """
    Effectue le split stratifié reproductible.
    """

    # ------------------------------------------------------------------------
    # Groupement par strate
    # ------------------------------------------------------------------------

    groups = defaultdict(list)

    for index, record in enumerate(records):
        key = stratification_key(record)
        groups[key].append(index)

    print()
    print("=== STRATIFICATION ===")
    print(f"Nombre de groupes : {len(groups)}")

    print()
    print("Strates du dataset source :")

    for key in sorted(groups):
        print(f"  {key}: {len(groups[key])}")

    # ------------------------------------------------------------------------
    # Calcul des quotas
    # ------------------------------------------------------------------------

    quotas, targets = compute_stratified_quotas(
        groups=groups,
        total_size=len(records),
    )

    print()
    print("Quotas par strate :")

    for key in sorted(quotas):
        q = quotas[key]

        print(
            f"  {key}: "
            f"train={q['train']}, "
            f"validation={q['validation']}, "
            f"test={q['test']}"
        )

    print()
    print("Totaux des quotas :")
    print(f"  Train       : {targets['train']}")
    print(f"  Validation  : {targets['validation']}")
    print(f"  Test        : {targets['test']}")

    # ------------------------------------------------------------------------
    # Allocation déterministe
    # ------------------------------------------------------------------------

    rng = random.Random(seed)

    train_indices = []
    validation_indices = []
    test_indices = []

    for key in sorted(groups):
        indices = list(groups[key])

        # Mélange reproductible de chaque strate.
        rng.shuffle(indices)

        q = quotas[key]

        train_end = q["train"]

        validation_end = (
            train_end
            + q["validation"]
        )

        train_indices.extend(
            indices[:train_end]
        )

        validation_indices.extend(
            indices[
                train_end:validation_end
            ]
        )

        test_indices.extend(
            indices[validation_end:]
        )

    # ------------------------------------------------------------------------
    # Mélange final reproductible de chaque split
    # ------------------------------------------------------------------------

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)

    # ------------------------------------------------------------------------
    # Vérifications
    # ------------------------------------------------------------------------

    if len(train_indices) != targets["train"]:
        raise RuntimeError(
            f"Train : {len(train_indices)} "
            f"au lieu de {targets['train']}"
        )

    if len(validation_indices) != targets["validation"]:
        raise RuntimeError(
            f"Validation : {len(validation_indices)} "
            f"au lieu de {targets['validation']}"
        )

    if len(test_indices) != targets["test"]:
        raise RuntimeError(
            f"Test : {len(test_indices)} "
            f"au lieu de {targets['test']}"
        )

    all_indices = (
        train_indices
        + validation_indices
        + test_indices
    )

    if len(all_indices) != len(records):
        raise RuntimeError(
            "Le nombre total d'indices ne correspond pas au dataset source."
        )

    if len(set(all_indices)) != len(all_indices):
        raise RuntimeError(
            "Un même record apparaît dans plusieurs splits."
        )

    if set(all_indices) != set(range(len(records))):
        raise RuntimeError(
            "Certains records du dataset source ne sont pas présents "
            "dans les splits."
        )

    train = [records[i] for i in train_indices]
    validation = [records[i] for i in validation_indices]
    test = [records[i] for i in test_indices]

    return train, validation, test, quotas


# ============================================================================
# CONTROLES DES SPLITS
# ============================================================================

def check_split_integrity(train, validation, test):
    print()
    print("=== CONTROLE D'INTEGRITE DES SPLITS ===")

    # ------------------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------------------

    train_ids = {
        record.get("id")
        for record in train
    }

    validation_ids = {
        record.get("id")
        for record in validation
    }

    test_ids = {
        record.get("id")
        for record in test
    }

    if train_ids & validation_ids:
        raise RuntimeError("Fuite d'IDs entre train et validation.")

    if train_ids & test_ids:
        raise RuntimeError("Fuite d'IDs entre train et test.")

    if validation_ids & test_ids:
        raise RuntimeError("Fuite d'IDs entre validation et test.")

    print("IDs : OK")

    # ------------------------------------------------------------------------
    # Contenu instruction + réponse
    # ------------------------------------------------------------------------

    train_qa = {
        qa_hash(record)
        for record in train
    }

    validation_qa = {
        qa_hash(record)
        for record in validation
    }

    test_qa = {
        qa_hash(record)
        for record in test
    }

    if train_qa & validation_qa:
        raise RuntimeError(
            "Fuite instruction/réponse entre train et validation."
        )

    if train_qa & test_qa:
        raise RuntimeError(
            "Fuite instruction/réponse entre train et test."
        )

    if validation_qa & test_qa:
        raise RuntimeError(
            "Fuite instruction/réponse entre validation et test."
        )

    print("Instruction + réponse : OK")

    # ------------------------------------------------------------------------
    # Conservation
    # ------------------------------------------------------------------------

    total = len(train) + len(validation) + len(test)

    print(f"Records après split : {total}")

    if total != 5000:
        raise RuntimeError(
            f"Conservation invalide : {total} au lieu de 5000."
        )

    print("Conservation des records : OK")


# ============================================================================
# DISTRIBUTIONS
# ============================================================================

def print_distribution(name, records):
    print()
    print(f"--- {name} ---")

    source_counter = Counter()
    language_counter = Counter()
    task_counter = Counter()
    strata_counter = Counter()

    for record in records:
        source = get_source_dataset(record)
        language = normalize(record.get("language"))
        task = normalize(record.get("task"))

        source_counter[source] += 1
        language_counter[language] += 1
        task_counter[task] += 1
        strata_counter[
            (source, language, task)
        ] += 1

    print("Sources :")
    for key in sorted(source_counter):
        print(f"  {key}: {source_counter[key]}")

    print("Langues :")
    for key in sorted(language_counter):
        print(f"  {key}: {language_counter[key]}")

    print("Tasks :")
    for key in sorted(task_counter):
        print(f"  {key}: {task_counter[key]}")

    print("Strates :")
    for key in sorted(strata_counter):
        print(f"  {key}: {strata_counter[key]}")


# ============================================================================
# VERIFICATION DES QUOTAS
# ============================================================================

def check_stratification(train, validation, test, quotas):
    print()
    print("=== CONTROLE DE LA STRATIFICATION ===")

    actual = {
        "train": Counter(
            stratification_key(record)
            for record in train
        ),
        "validation": Counter(
            stratification_key(record)
            for record in validation
        ),
        "test": Counter(
            stratification_key(record)
            for record in test
        ),
    }

    errors = []

    for key in sorted(quotas):
        expected_train = quotas[key]["train"]
        expected_validation = quotas[key]["validation"]
        expected_test = quotas[key]["test"]

        actual_train = actual["train"][key]
        actual_validation = actual["validation"][key]
        actual_test = actual["test"][key]

        if actual_train != expected_train:
            errors.append(
                (
                    key,
                    "train",
                    expected_train,
                    actual_train,
                )
            )

        if actual_validation != expected_validation:
            errors.append(
                (
                    key,
                    "validation",
                    expected_validation,
                    actual_validation,
                )
            )

        if actual_test != expected_test:
            errors.append(
                (
                    key,
                    "test",
                    expected_test,
                    actual_test,
                )
            )

    if errors:
        print("ERREURS DE STRATIFICATION :")

        for error in errors:
            print(
                f"  {error[0]} / {error[1]} : "
                f"attendu={error[2]}, obtenu={error[3]}"
            )

        raise RuntimeError(
            "La stratification obtenue ne correspond pas aux quotas."
        )

    print("Stratification : OK")


# ============================================================================
# ECRITURE
# ============================================================================

def write_jsonl(path, records):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================================
# CONTROLE DE REPRODUCTIBILITE
# ============================================================================

def split_fingerprint(train, validation, test):
    """
    Empreinte déterministe des trois splits.

    Utile pour vérifier qu'un nouveau lancement produit exactement
    les mêmes splits.
    """

    payload = {
        "train": [
            record.get("id")
            for record in train
        ],
        "validation": [
            record.get("id")
            for record in validation
        ],
        "test": [
            record.get("id")
            for record in test
        ],
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SPLIT STRATIFIÉ DU DATASET SFT")
    print("=" * 70)

    print()
    print(f"Entrée : {INPUT_FILE}")
    print(f"Seed   : {SEED}")

    # ------------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------------

    records = load_jsonl(INPUT_FILE)

    print(f"Records chargés : {len(records)}")

    if len(records) != 5000:
        raise RuntimeError(
            f"Le dataset source doit contenir 5000 records. "
            f"Trouvé : {len(records)}"
        )

    # ------------------------------------------------------------------------
    # Contrôles source
    # ------------------------------------------------------------------------

    validate_source_dataset(records)

    # ------------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------------

    train, validation, test, quotas = stratified_split(
        records=records,
        seed=SEED,
    )

    # ------------------------------------------------------------------------
    # Contrôles
    # ------------------------------------------------------------------------

    check_split_integrity(
        train,
        validation,
        test,
    )

    check_stratification(
        train,
        validation,
        test,
        quotas,
    )

    # ------------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------------

    print_distribution(
        "TRAIN",
        train,
    )

    print_distribution(
        "VALIDATION",
        validation,
    )

    print_distribution(
        "TEST",
        test,
    )

    # ------------------------------------------------------------------------
    # Empreinte reproductibilité
    # ------------------------------------------------------------------------

    fingerprint = split_fingerprint(
        train,
        validation,
        test,
    )

    # ------------------------------------------------------------------------
    # Ecriture
    # ------------------------------------------------------------------------

    write_jsonl(
        TRAIN_FILE,
        train,
    )

    write_jsonl(
        VALIDATION_FILE,
        validation,
    )

    write_jsonl(
        TEST_FILE,
        test,
    )

    # ------------------------------------------------------------------------
    # Résultat final
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT TERMINÉ")
    print("=" * 70)

    print()
    print(f"Train       : {len(train)}")
    print(f"Validation  : {len(validation)}")
    print(f"Test        : {len(test)}")
    print(f"Total       : {len(train) + len(validation) + len(test)}")

    print()
    print("Fichiers :")
    print(f"  Train       : {TRAIN_FILE}")
    print(f"  Validation  : {VALIDATION_FILE}")
    print(f"  Test        : {TEST_FILE}")

    print()
    print(f"Seed         : {SEED}")
    print(f"Fingerprint  : {fingerprint}")

    print()
    print("Reproductibilité : OK")
    print("Même dataset + même seed = même split.")


if __name__ == "__main__":
    main()
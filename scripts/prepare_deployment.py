from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "dpo_v8"
DEPLOY_DIR = ROOT / "deployment" / "models" / "dpo_v8"

REQUIRED_FILES = [
    "adapter_config.json",
]

def main():
    print("=== Préparation du déploiement P14 ===")

    if not MODEL_DIR.is_dir():
        sys.exit(f"ERREUR : modèle introuvable : {MODEL_DIR}")

    for filename in REQUIRED_FILES:
        if not (MODEL_DIR / filename).is_file():
            sys.exit(f"ERREUR : fichier manquant : {filename}")

    weights = [
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    if not any((MODEL_DIR / name).is_file() for name in weights):
        sys.exit("ERREUR : poids de l'adaptateur LoRA introuvables.")

    with (MODEL_DIR / "adapter_config.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        sys.exit("ERREUR : modèle de base absent de adapter_config.json.")

    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)

    DEPLOY_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MODEL_DIR, DEPLOY_DIR)

    manifest = {
        "base_model": base_model,
        "adapter": "dpo_v3",
        "adapter_type": config.get("peft_type"),
        "deployment": "Linux + Docker + vLLM",
        "adapter_path": "/models/dpo_v3",
    }

    manifest_path = ROOT / "deployment" / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Modèle de base : {base_model}")
    print(f"Adaptateur copié : {DEPLOY_DIR}")
    print(f"Manifeste : {manifest_path}")
    print("\nPréparation terminée.")
    print("Transfère le dossier deployment/ sur la VM Linux.")

if __name__ == "__main__":
    main()
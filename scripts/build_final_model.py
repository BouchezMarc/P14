from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "unsloth/Qwen3-1.7B-Base"
DPO_CHECKPOINT = Path("models/dpo_v8")
TOKENIZER_SOURCE = Path("models/sft_v3/checkpoint-1000")
OUTPUT_DIR = Path("models/final_model_dpo_v8")


def main():
    if not (DPO_CHECKPOINT / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adaptateur DPO introuvable : {DPO_CHECKPOINT}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} existe déjà. "
            "Choisis un nouveau dossier de sortie pour éviter de mélanger les poids."
        )

    print("1/5 - Chargement du modèle de base...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("2/5 - Chargement de l'adaptateur DPO...")
    model = PeftModel.from_pretrained(
        model,
        str(DPO_CHECKPOINT),
        is_trainable=False,
    )

    print("3/5 - Fusion des poids DPO dans le modèle de base...")
    model = model.merge_and_unload(safe_merge=True)

    print("4/5 - Sauvegarde du modèle fusionné...")
    OUTPUT_DIR.mkdir(parents=True)
    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print("5/5 - Sauvegarde du tokenizer SFT...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_SOURCE))
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModèle final sauvegardé dans : {OUTPUT_DIR}")
    print("Vérifie sa génération avant de l'utiliser en déploiement.")


if __name__ == "__main__":
    main()
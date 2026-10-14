from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "models" / "dpo_v8"
OUTPUT_PATH = ROOT / "models" / "dpo_v8_merged"

def main():
    if not (ADAPTER_PATH / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adaptateur introuvable : {ADAPTER_PATH}"
        )

    print("1. Lecture de la configuration LoRA...")
    import json
    config = json.loads(
        (ADAPTER_PATH / "adapter_config.json").read_text(
            encoding="utf-8"
        )
    )
    base_model = config["base_model_name_or_path"]
    print(f"Modèle de base : {base_model}")

    print("2. Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("3. Chargement du modèle de base sur CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("4. Chargement de l'adaptateur DPO...")
    model = PeftModel.from_pretrained(
        model,
        str(ADAPTER_PATH),
    )

    print("5. Fusion des poids...")
    model = model.merge_and_unload(safe_merge=True)

    print(f"6. Sauvegarde du modèle final : {OUTPUT_PATH}")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("Modèle final sauvegardé.")
    print(f"Répertoire : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
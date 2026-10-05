import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-1.7B-Base"


MODEL_PATHS = {
    "base": None,

    "sft_v1": "models/sft_v1",
    "sft_v2": "models/sft_v2",
    "sft_v3": "models/sft_v3",
    "sft_v4": "models/sft_v4",
    "sft_v5": "models/sft_v5",
    "sft_v6": "models/sft_v6",
    "sft_v7": "models/sft_v7",
    "sft_v8": "models/sft_8",
    "sft_v9": "models/sft_v9",
    "sft_v10": "models/sft_v10",
    "dpo_v1": "models/dpo_v1",
    "dpo_v2": "models/dpo_v2",
    "dpo_v3": "models/dpo_v3",
    "dpo_v4": "models/dpo_v4",
    "dpo_v5": "models/dpo_v5",
    "dpo_v6": "models/dpo_v6",
}


REWARDBENCH_FILE = "data/processed/medical_rewardbench.jsonl"

DEFAULT_OUTPUT_ROOT = "results/rewardbench"


def extract_answer(messages):

    if not isinstance(messages, list):
        return ""

    for message in messages:
        if message.get("role") == "assistant":
            return message.get("content", "")

    return ""


def load_model(model_name):

    adapter_path = MODEL_PATHS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

    model.eval()

    return model, tokenizer
def compute_log_probability(
    model,
    tokenizer,
    prompt,
    answer,
    device,
    max_seq_length,
):

    full_text = prompt + "\n" + answer

    prompt_tokens = tokenizer(
        prompt + "\n",
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )

    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)

    prompt_length = prompt_tokens["input_ids"].shape[1]

    if input_ids.shape[1] <= prompt_length:
        return float("-inf")

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        2,
        labels.unsqueeze(-1),
    ).squeeze(-1)


    answer_log_probs = token_log_probs[
        :,
        prompt_length-1:
    ]

    return answer_log_probs.mean().item()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_PATHS.keys()),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()


    dataset = load_dataset(
        "json",
        data_files=REWARDBENCH_FILE,
        split="train",
    )


    if args.start:
        dataset = dataset.select(
            range(
                args.start,
                len(dataset),
            )
        )


    if args.limit is not None:
        dataset = dataset.select(
            range(
                min(
                    args.limit,
                    len(dataset),
                )
            )
        )


    total = len(dataset)


    print("=" * 70)
    print("REWARDBENCH DPO PREFERENCE")
    print("=" * 70)
    print(f"Model    : {args.model}")
    print(f"Examples : {total}")


    model, tokenizer = load_model(
        args.model
    )


    device = next(
        model.parameters()
    ).device


    if args.output is None:

        output_dir = os.path.join(
            DEFAULT_OUTPUT_ROOT,
            args.model,
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        output_file = os.path.join(
            output_dir,
            "predictions_choosen.jsonl",
        )

    else:

        output_file = args.output

        output_dir = os.path.dirname(
            output_file
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )


    correct = 0
    processed = 0


    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as fout:


        for index, example in enumerate(dataset):


            chosen = extract_answer(
                example["chosen"]
            )

            rejected = extract_answer(
                example["rejected"]
            )


            prompt = example["prompt"]


            chosen_score = compute_log_probability(
                model,
                tokenizer,
                prompt,
                chosen,
                device,
                args.max_seq_length,
            )


            rejected_score = compute_log_probability(
                model,
                tokenizer,
                prompt,
                rejected,
                device,
                args.max_seq_length,
            )


            is_correct = (
                chosen_score > rejected_score
            )


            if is_correct:
                correct += 1


            processed += 1


            result = {
                "id": example.get("id"),
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "correct": is_correct,
            }


            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


            print(
                f"{processed}/{total} "
                f"| correct={correct}"
            )


    accuracy = (
        correct / processed
        if processed
        else 0
    )


    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Total     : {processed}"
    )

    print(
        f"Correct   : {correct}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Output    : {output_file}"
    )


if __name__ == "__main__":
    main()
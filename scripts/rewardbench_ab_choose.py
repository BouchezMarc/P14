import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL="Qwen/Qwen3-1.7B-Base"

MODEL_PATHS={
    "base":None,
    "sft_v1":"models/sft_v1",
    "sft_v2":"models/sft_v2",
    "sft_v3":"models/sft_v3",
    "sft_v4":"models/sft_v4",
    "sft_v5":"models/sft_v5",
    "dpo_v1":"models/dpo_v1",
    "dpo_v2":"models/dpo_v2",
    "dpo_v3":"models/dpo_v3",
}

REWARDBENCH_FILE="data/processed/medical_rewardbench.jsonl"
DEFAULT_OUTPUT_ROOT="results/rewardbench"
RANDOM_SEED=42

PROMPT_TEMPLATE="""You are evaluating two medical answers.

Choose the answer that is clinically better, safer and more appropriate.

Do not explain.
Do not provide reasoning.
Return ONLY one letter: A or B.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Your choice:
"""

import re

def parse_choice(text):
    if not text:
        return None

    text = text.upper()

    # Suppression globale des mots parasites
    text = re.sub(        
        r"(ANSWER|CHOICE|YOUR CHOICE|OR|[:<,.\s]|\\N)",
        "",
        text
    )

    if text in ("A", "B"):
        return text

    return None


def extract_answer(messages):
    if not isinstance(messages,list):
        return ""

    for message in messages:
        if message.get("role")=="assistant":
            return message.get("content","")

    return ""


def load_model(model_name):

    adapter_path=MODEL_PATHS[model_name]

    tokenizer=AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    tokenizer.padding_side="left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    model=AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if adapter_path:
        model=PeftModel.from_pretrained(
            model,
            adapter_path
        )

    model.eval()

    return model,tokenizer

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--model",required=True,choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--limit",type=int,default=None)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--max-seq-length",type=int,default=2048)
    parser.add_argument("--max-new-tokens",type=int,default=2)
    parser.add_argument("--output",default=None)

    args=parser.parse_args()

    #random.seed(RANDOM_SEED)

    dataset=load_dataset(
        "json",
        data_files=REWARDBENCH_FILE,
        split="train"
    )

    if args.start:
        dataset=dataset.select(
            range(args.start,len(dataset))
        )

    if args.limit is not None:
        dataset=dataset.select(
            range(min(args.limit,len(dataset)))
        )

    total=len(dataset)

    print("="*70)
    print("REWARDBENCH")
    print("="*70)
    print(f"Model : {args.model}")
    print(f"Examples : {total}")

    model,tokenizer=load_model(args.model)

    device=next(model.parameters()).device

    if args.output is None:
        output_dir=os.path.join(
            DEFAULT_OUTPUT_ROOT,
            args.model
        )
        os.makedirs(output_dir,exist_ok=True)

        output_file=os.path.join(
            output_dir,
            "predictions.jsonl"
        )
    else:
        output_file=args.output
        output_dir=os.path.dirname(output_file)

        if output_dir:
            os.makedirs(output_dir,exist_ok=True)


    correct=0
    valid=0
    invalid=0
    processed=0


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as fout:

        for batch_start in range(
            0,
            total,
            args.batch_size
        ):

            batch_end=min(
                batch_start+args.batch_size,
                total
            )

            batch=dataset.select(
                range(
                    batch_start,
                    batch_end
                )
            )

            prompts=[]
            metadata=[]

            for example in batch:

                chosen=extract_answer(
                    example["chosen"]
                )

                rejected=extract_answer(
                    example["rejected"]
                )

                rng = random.Random(
                    f"{RANDOM_SEED}_{example['id']}"
                )
                                
                if rng.random()<0.5:

                    answer_a=chosen
                    answer_b=rejected
                    expected="A"

                else:

                    answer_a=rejected
                    answer_b=chosen
                    expected="B"


                prompts.append(
                    PROMPT_TEMPLATE.format(
                        question=example["prompt"],
                        answer_a=answer_a,
                        answer_b=answer_b
                    )
                )

                metadata.append(
                    {
                        "id":example["id"],
                        "expected":expected
                    }
                )


            inputs=tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_length
            )

            inputs={
                k:v.to(device)
                for k,v in inputs.items()
            }


            with torch.inference_mode():

                outputs=model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )


            input_length=inputs["input_ids"].shape[1]


            for i,output in enumerate(outputs):

                generated=output[input_length:]

                raw_output=tokenizer.decode(
                    generated,
                    skip_special_tokens=True
                ).strip()

                decision=parse_choice(
                    raw_output
                )

                expected=metadata[i]["expected"]


                if decision is None:

                    invalid+=1

                else:

                    valid+=1

                    if decision==expected:
                        correct+=1


                processed+=1


                result={
                    "id":metadata[i]["id"],
                    "decision":decision,
                    "expected":expected,
                    "correct":(
                        decision==expected
                        if decision
                        else False
                    ),
                    "raw_output":raw_output
                }


                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    )+"\n"
                )


                print(
                    f"{processed}/{total} "
                    f"| valid={valid} "
                    f"| invalid={invalid} "
                    f"| correct={correct}"
                )


    accuracy = correct / valid if valid else 0


    print()
    print("="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Total     : {total}")
    print(f"Valid     : {valid}")
    print(f"Invalid   : {invalid}")
    print(f"Correct   : {correct}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Output    : {output_file}")


if __name__=="__main__":
    main()
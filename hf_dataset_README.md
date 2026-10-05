\# P14 Datasets



Datasets prepared for the P14 medical AI proof-of-concept.



\## Objective



This repository contains the processed datasets used to train and evaluate the P14 medical question-answering pipeline.



The project is a technical proof of concept. The datasets and models are \*\*not clinically validated\*\* and must not be used for medical decision-making.



\## Dataset sources



The SFT dataset was constructed from three public medical question-answering datasets.



| Source                                                                  | Use in P14                 | Language | License    |

| ----------------------------------------------------------------------- | -------------------------- | -------- | ---------- |

| \[FrenchMedMCQA](https://huggingface.co/datasets/qanastek/frenchmedmcqa) | Medical multiple-choice QA | French   | Apache-2.0 |

| \[MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl)          | Medical QA / MCQ           | French   | CC BY 4.0  |

| \[MedQuAD](https://github.com/abachaa/MedQuAD)                           | Medical question-answering | English  | CC BY 4.0  |



The original licenses and attribution requirements remain applicable to the corresponding source material.



\## SFT dataset



The final SFT corpus contains 5,000 anonymized examples.



Composition:



\* FrenchMedMCQA: 1,667 examples

\* MediQAl: 1,666 examples

\* MedQuAD: 1,667 examples

\* Total: 5,000 examples



The final processed corpus is:



`sft/sft\_5000\_anonymized.jsonl`



\### Reproducible split



The SFT dataset was split using seed `3407`.



| Split      | Examples |

| ---------- | -------: |

| Train      |    4,000 |

| Validation |      500 |

| Test       |      500 |

| Total      |    5,000 |



Files:



\* `sft/sft\_train.jsonl`

\* `sft/sft\_validation.jsonl`

\* `sft/sft\_test.jsonl`



The test set is kept separate from training and validation data.



\## Anonymization and processing



The SFT corpus was processed before training.



Processing includes:



\* normalization of source records;

\* conversion to a common JSONL format;

\* anonymization of the prepared corpus;

\* construction of a unified instruction/response representation;

\* reproducible train/validation/test splitting.



Processing version:



`v0.1`



The processed dataset should not be interpreted as a replacement for the original source datasets.



\## DPO dataset



The DPO training data contained preference pairs in `chosen` / `rejected` format.



The dataset in:



\* `dpo/dpo\_train.jsonl`

\* `dpo/dpo\_validation.jsonl`



corresponds to the dataset version used to train the P14 `dpo\_v3` model.



The source directory in the project was:



`data/processed/dataset\_dpo\_v2/`



The directory name refers to the dataset-generation version and should not be confused with the resulting model version `dpo\_v3`.



\## RewardBench



`rewardbench/medical\_rewardbench.jsonl` contains the medical preference benchmark used for independent evaluation.



The benchmark contains 776 evaluation examples in the current P14 evaluation.



RewardBench evaluation is used to measure preference selection performance. It is \*\*not clinical validation\*\*.



\## Data format



All datasets are stored as UTF-8 JSONL files.



The processed datasets are intended to be directly usable with the Hugging Face `datasets` library and the P14 training/evaluation scripts.



\## Reproducibility



The P14 pipeline is designed to be reproducible.



Important parameters include:



\* SFT dataset: 5,000 examples

\* SFT split seed: `3407`

\* SFT split: 80/10/10

\* SFT training examples: 4,000

\* SFT validation examples: 500

\* SFT test examples: 500

\* DPO dataset: `dataset\_dpo\_v2`

\* DPO model evaluated: `dpo\_v3`



The corresponding training code, configuration and environment are maintained in the P14 project repository.



\## Limitations



The datasets primarily represent medical question-answering and examination-style tasks. They are not a representative clinical triage dataset.



In particular:



\* medical QA performance does not establish clinical triage performance;

\* benchmark performance does not establish clinical safety;

\* source datasets may contain biases related to examination or educational content;

\* generated or fine-tuned model outputs require independent clinical evaluation before any real-world clinical use.



\## Attribution



When redistributing or using material derived from the source datasets, users must comply with the applicable original licenses and attribution requirements.



\### FrenchMedMCQA



FrenchMedMCQA is distributed under the Apache License 2.0.



Source:

https://huggingface.co/datasets/qanastek/frenchmedmcqa



\### MediQAl



MediQAl is distributed under the Creative Commons Attribution 4.0 International license (CC BY 4.0).



Source:

https://huggingface.co/datasets/ANR-MALADES/MediQAl



\### MedQuAD



MedQuAD is distributed under the Creative Commons Attribution 4.0 International license (CC BY 4.0).



Source:

https://github.com/abachaa/MedQuAD



Original MedQuAD paper:



Ben Abacha, A. and Demner-Fushman, D. (2019), "A Question-Entailment Approach to Question Answering", BMC Bioinformatics.



\## Status



P14 dataset release for the technical proof-of-concept.



This repository is intended for research, experimentation and reproducibility.




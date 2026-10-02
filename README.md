# P14 — Medical AI Triage POC

Prototype de fine-tuning et d’évaluation d’un modèle de langage compact pour le **question-réponse médical**, dans le cadre du projet CHSA.

Le projet étudie une chaîne de spécialisation progressive :

**Qwen3-1.7B-Base → SFT LoRA → DPO LoRA → évaluation**

L’objectif est de mesurer de manière reproductible l’effet du fine-tuning supervisé et de l’optimisation par préférences sur un modèle compact, avant toute considération de déploiement dans un contexte de triage.

> **Important : ce projet est un POC de recherche et d’ingénierie. Les résultats obtenus ne constituent pas une validation clinique et le modèle n’est pas destiné à une utilisation médicale réelle.**
---
## 1. Objectifs

Le projet P14 vise à évaluer expérimentalement si un petit modèle de langage peut être spécialisé sur des données médicales francophones et anglophones au moyen de :
1. **Supervised Fine-Tuning (SFT)**    
2. **Direct Preference Optimization (DPO)**    
3. **Évaluation automatique structurée par un modèle juge**    
4. **Évaluation indépendante sur un benchmark de préférences médicales**    

Le modèle de base retenu est :
```text
Qwen/Qwen3-1.7B-Base
```

Le choix d’un modèle de 1,7 milliard de paramètres permet de réaliser les expérimentations localement sur une machine équipée d’une RTX 5080 Laptop GPU de 16 Go de VRAM.

---
# 2. Pipeline
```text
                         ┌─────────────────────┐
                         │  Medical datasets   │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Anonymisation /     │
                         │ preprocessing       │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ SFT dataset         │
                         │ 5000 examples       │
                         └──────────┬──────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │ Qwen3-1.7B-Base              │
                    └──────────────┬───────────────┘
                              LoRA / PEFT
                                   ▼
                    ┌──────────────────────────────┐
                    │ SFT model                    │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ DPO preference dataset       │
                    └──────────────┬───────────────┘
                              LoRA / PEFT
                                   ▼
                    ┌──────────────────────────────┐
                    │ DPO model                    │
                    └──────────────┬───────────────┘
                    ┌──────────────┴───────────────┐
                    ▼                              ▼
          ┌────────────────────┐        ┌─────────────────────┐
          │ Medical QA         │        │ Medical RewardBench │
          │ evaluation         │        │ pairwise evaluation │
          └─────────┬──────────┘        └──────────┬──────────┘
                    ▼                              ▼
          ┌────────────────────┐        ┌─────────────────────┐
          │ Ministral judge    │        │ chosen / rejected   │
          │                    │        │ preference accuracy │
          └────────────────────┘        └─────────────────────┘
```

---
# 3. Modèle

## Modèle de base
```text
Qwen/Qwen3-1.7B-Base
```

Le modèle de base est utilisé comme référence expérimentale.

Les modèles SFT et DPO sont des adaptations **LoRA / PEFT** du modèle de base et non des modèles entièrement réentraînés.

### Paramètres principaux
```text
Base model       : Qwen/Qwen3-1.7B-Base
Fine-tuning      : LoRA
Quantification   : non requise pour le protocole retenu
Max sequence     : 2048 tokens
SFT loss         : completion-only loss
```

Le protocole utilise **LoRA**, conformément à l'énoncé du projet. QLoRA n'est pas utilisé comme protocole de référence.

---
# 4. Données

## Dataset SFT

Le dataset principal est :
```text
data/processed/sft_5000_anonymized.jsonl
```

Il contient environ 5000 exemples médicaux préparés pour le fine-tuning supervisé.

La composition retenue est :

|Source|Nombre|
|---|--:|
|FrenchMedMCQA|1667|
|MediQAl|1666|
|MedQuAD|1667|
|**Total**|**5000**|

Les données sont réparties de manière reproductible en :

|Split|Nombre|
|---|--:|
|Train|4000|
|Validation|500|
|Test|500|

Le split est stratifié afin de conserver la distribution des sources tout en garantissant une procédure reproductible.

---
# 5. Données DPO

Le DPO utilise des paires de préférences :
```text
prompt
chosen
rejected
```

Les données de préférence sont séparées des données utilisées pour l'évaluation finale afin d'éviter de mélanger apprentissage et mesure.

Les fichiers principaux sont :
```text
data/processed/dpo_train.jsonl
data/processed/dpo_validation.jsonl
```

Le pipeline DPO part du modèle SFT :
```text
Qwen3-1.7B-Base
        ▼
      SFT
        ▼
   SFT adapter
        ▼
      DPO
        ▼
   DPO adapter
```
---
# 6. RewardBench médical

Un benchmark indépendant est conservé dans :
```text
data/processed/medical_rewardbench.jsonl
```

Il contient :
```text
id
prompt
chosen
rejected
label_type
quality
provenance
```

Les labels ont été préparés avec une information d'expertise :
```text
expert_reviewed_labels = true
clinical_validation_status = expert_reviewed_labels
```

Le dataset provient du corpus de préférences médicales utilisé dans le pipeline et a été dédoublonné pour obtenir :
```text
776 exemples uniques
```

Répartition :
```text
easy    : 237
hard    : 196
length  : 180
human   : 163
```

### Rôle du RewardBench

RewardBench est utilisé comme **évaluation indépendante de préférences**.

Pour chaque modèle, le système mesure sa capacité à préférer la réponse `chosen` à la réponse `rejected`.

Cette mesure est rapportée séparément des scores de question-réponse.
> La présence de labels revus par des experts ne signifie pas que le benchmark constitue à lui seul une validation clinique du système.
---
# 7. Évaluation Medical QA

L'évaluation principale compare :
```text
Base
SFT
DPO
```
sur un jeu de questions médicales séparé des données d'entraînement.

Chaque modèle génère une réponse à partir du même prompt.

Les sorties sont ensuite évaluées par un modèle juge.

Le juge utilisé pour le protocole actuel est :
```text
ministral-14b-2512
```

avec notamment :
```text
reasoning_effort = medium
max_tokens = 500
response_format = JSON
```
---
## Critères d'évaluation

L'évaluation utilise plusieurs dimensions :
### Faithfulness
La réponse respecte-t-elle les informations fournies et évite-t-elle les affirmations non justifiées ?

### Relevance
La réponse répond-elle directement à la question posée ?

### Correctness
Les informations médicales données sont-elles correctes par rapport à la réponse de référence ?

### Completeness
Les éléments importants attendus sont-ils présents ?

### Overall
Score global produit à partir de l'évaluation du juge.

---
# 8. Pourquoi ne pas utiliser uniquement Exact Match / F1 ?

Les réponses médicales sont génératives et peuvent être formulées de nombreuses façons tout en exprimant la même information.

Par conséquent :
```text
Exact Match
Token F1
```
ne sont pas considérés comme les métriques principales du projet.

Ils peuvent néanmoins être conservés comme métriques complémentaires.

L'évaluation principale repose sur l'analyse structurée de :
```text
faithfulness
relevance
correctness
completeness
```
afin de mieux caractériser les différences entre les modèles.

---
# 9. Résultats expérimentaux

Les expérimentations ont notamment porté sur :
```text
Base
SFT
DPO
```

avec plusieurs versions intermédiaires :
```text
models/sft_v1
models/sft_v2
models/sft_v3
models/sft_v4
models/sft_v5

models/dpo_v1
models/dpo_v2
models/dpo_v3
```

Les adapters sont chargés au-dessus de :
```text
Qwen/Qwen3-1.7B-Base
```
## Tendances observées

Les évaluations montrent une évolution différente selon les critères.

À titre indicatif, les expérimentations récentes donnent notamment :

|Critère|Base|SFT|DPO|
|---|--:|--:|--:|
|Relevance|~2.65|~2.93|~2.95|
|Correctness|~2.30|~2.12|~2.06–2.14|
|Faithfulness|~2.09|~1.89|~1.89–1.98|
|Completeness|~1.72|~1.67–1.76|~1.67–1.76|

Les valeurs exactes dépendent de la version du modèle et du jeu évalué.

Ces résultats montrent notamment que l'amélioration de la **relevance** ne s'accompagne pas automatiquement d'une amélioration de la **correctness** ou de la **faithfulness**.

Il s'agit d'un résultat expérimental important du POC : le fine-tuning ne doit donc pas être considéré comme automatiquement bénéfique sur l'ensemble des dimensions d'évaluation.

---
# 10. RewardBench

L'évaluation RewardBench est effectuée séparément.

Le principe est :
```text
prompt
   │
   ├── chosen
   └── rejected
```

Le modèle attribue une probabilité à chaque réponse.

La décision est ensuite comparée au label de préférence du benchmark.

La métrique principale est :
```text
preference accuracy
```

Exemple de résultat actuellement observé pour une évaluation DPO :
```text
Total     : 776
Correct   : 299
Accuracy  : 0.3853
```

Cette mesure doit être interprétée comme une **mesure de préférence sur ce benchmark**, et non comme une mesure directe de qualité clinique.

---
# 11. Anonymisation et conformité des données

Le pipeline utilise un dataset anonymisé destiné aux expérimentations.

Les traitements de données doivent conserver les informations de provenance nécessaires à l'audit :
```text
source
provenance
quality
processing_version
clinical_context
source_specific
```
Les documents relatifs au RGPD et aux traitements de données font partie du périmètre du projet.

Le modèle entraîné dans ce dépôt ne doit pas être considéré comme autorisé à traiter des données patients réelles simplement parce que le dataset d'entraînement a été anonymisé.

---
# 12. Environnement

Le développement et les entraînements actuels sont réalisés sous Windows.

### Matériel principal
```text
GPU  : NVIDIA GeForce RTX 5080 Laptop GPU
VRAM : 16 GB
RAM  : 32 GB
```
### Environnement logiciel
```text
OS          : Windows
Python      : 3.14.7
uv          : gestionnaire d'environnement
PyTorch     : 2.14.0 + CUDA 13.0
Transformers: 5.17.0
TRL         : 0.24.0
PEFT        : 0.21.0
bitsandbytes: 0.50.2
Unsloth     : 2025.7.2
```
Le projet utilise l'index PyTorch CUDA approprié à l'environnement local.

---
# 13. Installation

Le projet utilise `uv`.

Depuis la racine du dépôt :
```bash
uv sync
```

Puis vérifier l'environnement CUDA :
```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Le résultat attendu est notamment :
```text
torch.cuda.is_available() == True
```

avec une RTX 5080 comme GPU détecté.

---
# 14. Structure du projet

Structure indicative :
```text
projet14/
│
├── data/
│   ├── raw/
│   │
│   └── processed/
│       ├── sft_5000_anonymized.jsonl
│       ├── sft_train.jsonl
│       ├── sft_validation.jsonl
│       ├── sft_test.jsonl
│       ├── dpo_train.jsonl
│       ├── dpo_validation.jsonl
│       └── medical_rewardbench.jsonl
│
├── models/
│   ├── sft_v1/
│   ├── sft_v2/
│   ├── sft_v3/
│   ├── sft_v4/
│   ├── sft_v5/
│   ├── dpo_v1/
│   ├── dpo_v2/
│   └── dpo_v3/
│
├── results/
│   ├── sft/
│   ├── dpo/
│   └── rewardbench/
│
├── scripts/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
│
├── pyproject.toml
├── uv.lock
└── README.md
```
La structure exacte peut évoluer avec les versions expérimentales.

---
# 15. Reproductibilité

La reproductibilité est une exigence du projet.

Les éléments suivants doivent être fixés :
- seed ;    
- versions des dépendances ;    
- modèle de base ;    
- dataset utilisé ;    
- split train/validation/test ;    
- paramètres LoRA ;    
- longueur maximale ;    
- paramètres de génération ;    
- prompt d'entraînement ;    
- prompt d'évaluation ;    
- modèle juge ;    
- paramètres du juge.    

Le split SFT utilise notamment :
```text
seed = 3407
```

avec :
```text
train      : 4000
validation : 500
test       : 500
```

Les expériences doivent être exécutées à partir des fichiers de données générés plutôt que de recréer implicitement les splits à chaque entraînement.

---
# 16. Principes d'évaluation

Pour éviter de confondre les différentes étapes du pipeline :
```text
TRAINING DATA
      │
      ├── SFT
      │
      └── DPO
           
TEST DATA
      │
      └── Medical QA evaluation
              │
              └── Ministral judge

REWARDBENCH
      │
      └── Pairwise preference evaluation
```
Les données utilisées pour l'apprentissage ne doivent pas être utilisées pour présenter une performance finale.

De même, le score RewardBench ne doit pas être fusionné avec les scores de l'évaluation QA.

---
# 17. Limites
Ce POC présente plusieurs limites importantes.

### Taille du modèle
Qwen3-1.7B est un modèle compact. Sa capacité est nécessairement limitée par rapport à des modèles médicaux ou généralistes beaucoup plus importants.

### Taille des données
Le SFT utilise environ 5000 exemples. Cette quantité permet d'étudier le pipeline mais reste limitée pour une spécialisation médicale robuste.

### Évaluation par LLM-as-a-Judge
L'évaluation utilisant Ministral constitue une mesure automatisée.
Elle ne remplace pas une évaluation réalisée par des professionnels de santé selon un protocole clinique contrôlé.

### Biais du benchmark
Les résultats dépendent de la distribution et de la qualité des datasets utilisés.

### Triage
Le projet est motivé par un cas d'usage de triage, mais les expérimentations actuelles portent principalement sur du **medical question-answering**.
Une bonne performance sur le QA ne constitue donc pas une démonstration de sécurité ou de performance pour un système de triage clinique.

### Génération
Les modèles peuvent produire des réponses incomplètes, incorrectes ou non suffisamment justifiées.
L'objectif du POC est précisément de mesurer ces phénomènes plutôt que de supposer que le fine-tuning les élimine.

---
# 18. Interprétation des résultats

Le projet ne considère pas :
```text
SFT > Base
```

ou :
```text
DPO > SFT
```

comme des hypothèses automatiquement vérifiées.

Chaque étape doit être évaluée sur plusieurs dimensions.

Une amélioration sur une métrique peut être accompagnée d'une dégradation sur une autre.

L'analyse finale doit donc présenter séparément :
```text
Relevance
Correctness
Faithfulness
Completeness
RewardBench preference accuracy
```

plutôt que de réduire le comportement du modèle à un score unique.

---
# 19. Statut du projet

**Statut : POC expérimental**

Le pipeline technique principal est opérationnel :
```text
Dataset
   ↓
SFT
   ↓
DPO
   ↓
Medical QA evaluation
   ↓
RewardBench evaluation
```
Les expériences permettent désormais de comparer directement le modèle de base, le modèle SFT et les différentes versions DPO.

Les résultats actuels montrent que le fine-tuning modifie effectivement le comportement du modèle, mais qu'une amélioration sur une dimension donnée ne garantit pas une amélioration globale de la qualité médicale.

Le travail restant consiste principalement à consolider les évaluations, documenter les expériences finales et interpréter les résultats dans les limites méthodologiques du POC.

---
# 20. Avertissement

Ce dépôt est destiné à la **recherche, à l'expérimentation et à l'évaluation de modèles de langage**.

Il ne constitue pas :
- un dispositif médical ;    
- un outil de diagnostic ;    
- un système de triage clinique validé ;    
- une recommandation médicale ;    
- une preuve de sécurité clinique.    

Aucune décision concernant un patient ne doit être prise sur la base des sorties de ce modèle.

---
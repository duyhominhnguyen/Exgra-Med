# EXGRA-MED: Extended Context Graph Alignment for Medical Vision-Language Models

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/pdf/2410.02615v3)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/MERGE-Group)
[![License](https://img.shields.io/github/license/your-username/exgra-med)](./LICENSE)
[![Website](https://img.shields.io/badge/Demo-Live-green)](https://your-demo-site.com) <!-- Optional -->

---

## Abstract

State-of-the-art medical multi-modal LLMs (med-MLLMs), such as LLAVA-MED and BIOMEDGPT, primarily depend on scaling model size and data volume, with training driven largely by autoregressive objectives.   However, we reveal that this approach can lead to weak vision-language alignment, making these models overly dependent on costly instruction-following data.  

To address this, we introduce **EXGRA-MED**, a novel multi-graph alignment framework that jointly aligns images, instruction responses, and extended captions in the latent space, advancing semantic grounding and cross-modal coherence.   To scale to large LLMs (e.g., LLaMa-7B), we develop an efficient end-to-end training scheme using black-box gradient estimation, enabling fast and scalable optimization.  

## 🏆 **Key Results**

:white_check_mark: **Reveals the data inefficiency of autoregressive modeling** — LLaVA-Med exhibits a significant performance drop when pre-trained on limited data, even after full fine-tuning on downstream tasks.

:white_check_mark: **Matches LLaVA-Med’s performance on Medical VQA** using only **10% of the pre-training data**, demonstrating the data efficiency of EXGRA-MED.

:white_check_mark: **Surpasses several SOTA medical multi-modal LLMs** when pre-trained on the full PMC-15M dataset (100%) with LLaMA-7B, across diverse tasks:
- (i) Medical Visual Question Answering (VQA)  
- (ii) Medical Visual Chatbot  
- (iii) Zero-shot Image Classification (as a VQA task)

---

## 🚨 News

- **[Jun 2025]** 🔓 Initial codebase release (preprocessing + VQA fine-tuning).
- **[Jun 2025]** 🧩 Checkpoints for EXGRA-MED + DCI and three VQA fine-tuned models now available.
- **[Jun 2025]** 📊 Evaluation scripts and demo for data-efficiency benchmark now online.

---

## 📦 Model Checkpoints

| Model                                  | Description                                | Download Link |
|----------------------------------------|--------------------------------------------|---------------|
| `llava-med`                            | LLaVa-Med (10% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med`                            | ExGra-Med (10% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med`                            | Our base EXGRA-MED model (100% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med`                            | Our base EXGRA-MED model (100% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med-dci`                        | EXGRA-MED + DCI-enhanced version           | [Link](#)     |
| `exgra-med-dci-vqa-rad`               | Fine-tuned on VQA-RAD                      | [Link](#)     |
| `exgra-med-dci-slake`                 | Fine-tuned on SLAKE                        | [Link](#)     |
| `exgra-med-dci-pathvqa`               | Fine-tuned on PATH-VQA                     | [Link](#)     |

---

## 🛠️ Installation

```bash
git clone https://github.com/your_username/exgra-med.git
cd exgra-med
conda create -n exgra-med python=3.10
conda activate exgra-med
pip install -r requirements.txt
```

-----
## Project Structure
```
exgra-med/
├── configs/                   # Training and model configs
├── data/                      # Preprocessing scripts and data utils
├── models/                    # Core EXGRA-MED & DCI architecture
├── scripts/                   # Shell scripts for training and evaluation
├── utils/                     # Miscellaneous helpers
└── README.md

```


-----
## Fine-tuning on VQA Tasks
We provide ready-to-use scripts to fine-tune **EXGRA-MED** and **EXGRA-MED + DCI** on three popular medical VQA benchmarks: **VQA-RAD**, **SLAKE**, and **PATH-VQA**.

Each script uses one of our pretrained checkpoints as the starting point.  👉 **Before running**, make sure to update the `--pretrained_path` in each `.sh` file to point to the correct location of the downloaded model.

```
# Example: Fine-tune on VQA-RAD
bash scripts/finetune_vqa_rad.sh

# Fine-tune on SLAKE
bash scripts/finetune_slake.sh

# Fine-tune on PATH-VQA
bash scripts/finetune_pathvqa.sh
```

-----
## Evaluation
You can run evaluation for each of the three key tasks:

## 1. Medical VQA Evaluation

```
bash scripts/eval_vqa.sh  # supports VQA-RAD, SLAKE, PATH-VQA
```



# EXGRA-MED: Extended Context Graph Alignment for Medical Vision-Language Models

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/pdf/2410.02615v3)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/MERGE-Group)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://github.com/duyhominhnguyen/Exgra-Med/blob/main/LICENSE)
[![Website](https://img.shields.io/badge/🌐%20Project%20Page-ExGra--Med-green)](https://exgra-med.github.io/)




<!--
## Collaborating Institutions

This project is a joint research effort between the following institutions:

<p align="left">
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/Stanford_logo.png" height="80" alt="Stanford University" title="Stanford University"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/eth-zurich_logo.jpg" height="120" alt="ETH Zurich" title="ETH Zurich"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/UCSD_logo.png" height="80" alt="UCSD" title="UCSD"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/Universita%CC%88t_Stuttgart_Logo.png" height="50" alt="unistuttgart" title="University of Stuttgart"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/imprs-is-logo.jpg" height="30" alt="imprs-is" title="imprs-is"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/mpi_logo.jpg" height="80" alt="mpi" title="mpi"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/dfki_logo.png" height="60" alt="DFKI" title="DFKI"/>
</p>

-->
---

## Abstract

State-of-the-art medical multi-modal LLMs (med-MLLMs), such as LLAVA-MED and BIOMEDGPT, primarily depend on scaling model size and data volume, with training driven largely by autoregressive objectives.   However, we reveal that this approach can lead to weak vision-language alignment, making these models overly dependent on costly instruction-following data.  

To address this, we introduce **EXGRA-MED, a novel multi-graph alignment framework that jointly aligns images, instruction responses, and extended captions** in the latent space, advancing semantic grounding and cross-modal coherence.   To scale to large LLMs (e.g., LLaMa-7B), we develop an efficient end-to-end training scheme using black-box gradient estimation, enabling fast and scalable optimization.  

<p align="center">
<img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/exgramed_v1.jpg" alt="Alt text" width="1400"/>
</p>


## 🏆 **Key Results**

:white_check_mark: **Reveals the data inefficiency of autoregressive modeling** — LLaVA-Med exhibits a significant performance drop when pre-trained on limited data, even after full fine-tuning on downstream tasks.

:white_check_mark: **Matches LLaVA-Med’s performance on Medical VQA** using only **10% of the pre-training data**, demonstrating the data efficiency of EXGRA-MED.

:white_check_mark: **Surpasses several SOTA medical multi-modal LLMs** when pre-trained on the full PMC-15M dataset (100%) with LLaMA-7B, across diverse tasks:
- (i) Medical Visual Question Answering (VQA)  
- (ii) Medical Visual Chatbot  
- (iii) Zero-shot Image Classification (as a VQA task)


<p align="center">
<img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/figures/exgra_med_result.png" alt="Alt text" width="600"/>
</p>


---
## Table of Contents

- [📣 News](#-news)
- [📦 Model Checkpoints](#-model-checkpoints)
- [🛠️ Installation](#installation)
- [📂 Project Structure](#project-structure)
- [📄 Dataset Configuration Files](#-dataset-configuration-files)
- [🔧 Fine-tuning on VQA Tasks](#-fine-tuning-on-vqa-tasks)
- [📈 Evaluation](#-evaluation)
  - [1. Medical VQA Evaluation](#1-medical-vqa-evaluation)
  - [2. Medical Visual Chatbot](#2-medical-visual-chatbot)
  - [3. Zero-shot Image Classification](#3-zero-shot-image-classification)
- [🔬 Data Efficiency Demonstration (10% vs 40%)](#-data-efficiency-demonstration-10-vs-40)
- [📖 Citation](#citation)

---
## 📣 News

- **[Jun 2025]** 🔓 Initial codebase release (preprocessing + VQA fine-tuning).
- **[Jun 2025]** 🧩 Checkpoints for EXGRA-MED + DCI and three VQA fine-tuned models now available.
- **[Jun 2025]** 📊 Evaluation scripts and demo for data-efficiency benchmark now online.

---

## 📦 Model Checkpoints

| Model                                  | Description                                | Download Link |
|----------------------------------------|--------------------------------------------|---------------|
| `llava-med-10`                            | LLaVa-Med (10% pre-trained PMC-15M)                   | [Link](#)     |
| `llava-med-40`                            | LLaVa-Med (40% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med-10`                            | ExGra-Med (10% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med-40`                            | ExGra-Med (40% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med`                            | ExGra-Med (10% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med`                            | Our base EXGRA-MED model (100% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med`                            | Our base EXGRA-MED model (100% pre-trained PMC-15M)                   | [Link](#)     |
| `exgra-med-dci`                        | EXGRA-MED + DCI-enhanced version           | [Link](#)     |
| `exgra-med-dci-vqa-rad`               | Fine-tuned on VQA-RAD                      | [Link](#)     |
| `exgra-med-dci-slake`                 | Fine-tuned on SLAKE                        | [Link](#)     |
| `exgra-med-dci-pathvqa`               | Fine-tuned on PATH-VQA                     | [Link](#)     |

---

## Installation

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
--------
## 📄 Dataset Configuration Files
We provide pre-built `.json` configuration files for all datasets used in VQA training and evaluation. These files specify paths, splits, and preprocessing parameters necessary for seamless execution.

| Dataset      | Task       | Config File Description       | Download Link              |
| ------------ | ---------- | ----------------------------- | -------------------------- |
| VQA-RAD      | VQA        | Train/val splits, QA pairs    | [vqa\_rad\_config.json](#) |
| SLAKE        | VQA        | Train/val splits, QA pairs    | [slake\_config.json](#)    |
| PATH-VQA     | VQA        | Train/val splits, QA pairs    | [pathvqa\_config.json](#)  |

To download our langauge-image multimodal instruction-folllowing dataset, please run the following script:

```
sh download_data.sh
```


 🔗 **Instructions**:

- Place the downloaded `.json` files under the `configs/datasets/` directory.

- Update file paths inside if needed to match your local dataset locations.

-----
## 🔧 Fine-tuning on VQA Tasks
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
## 📈 Evaluation
You can run evaluation for each of the three key tasks:

## 1. Medical VQA Evaluation

```
bash scripts/eval_vqa.sh  # supports VQA-RAD, SLAKE, PATH-VQA
```

## 2. Medical Visual Chatbot
```
bash scripts/eval_chatbot.sh
```

## 3. Zero-shot Image Classification
By reformulating image classification as visual question answering, we can generate predictions by solving the VQA task with multiple-choice questions.
```
bash scripts/eval_zero_shot.sh
```

------
## 🔬 Data Efficiency Demonstration (10% vs 40%)
To replicate our findings on LLAVA-MED’s data inefficiency and the strength of EXGRA-MED with 10% and 40% data (Tables 1 & 2 in the paper):


```
# Fine-tune EXGRA-MED with 10%/40% data on VQA task 
bash scripts/train_10percent.sh
```

```
# Fine-tune checkpoint LLaVa-Med with 10%/40% data on VQA task
bash scripts/train_llava_full.sh
```

## Citation
If you find this work useful, please cite our paper:

```
@article{nguyen2025exgra,
  title={EXGRA-MED: Extended Context Graph Alignment for Medical Vision- Language Models},
  author={Duy M. H. Nguyen, Nghiem T. Diep, Trung Q. Nguyen, Hoang-Bao Le, Tai Nguyen, Tien Nguyen, TrungTin Nguyen, Nhat Ho, Pengtao Xie, Roger Wattenhofer, James Zou, Daniel Sonntag, Mathias Niepert},
  journal={arXiv preprint arXiv:2410.02615},
  year={2025}
}
```

## Usage and License Notices: 
The data, code, and model checkpoints are intended and licensed for research use only. They are also subject to additional restrictions dictated by the Terms of Use: LLaMA, Vicuna and GPT-4 respectively. The data is made available under CC BY NC 4.0. The data, code, and model checkpoints may be used for non-commercial purposes and any models trained using the dataset should be used only for research purposes. It is expressly prohibited for models trained on this data to be used in clinical care or for any clinical decision making purposes.


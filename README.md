# EXGRA-MED: Extended Context Graph Alignment for Medical Vision-Language Models

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/pdf/2410.02615v3)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/MERGE-Group)
[![License](https://img.shields.io/github/license/your-username/exgra-med)](./LICENSE)
[![Website](https://img.shields.io/badge/Demo-Live-green)](https://your-demo-site.com) <!-- Optional -->

---

## Abstract

State-of-the-art medical multi-modal LLMs (med-MLLMs), such as LLAVA-MED and BIOMEDGPT, primarily depend on scaling model size and data volume, with training driven largely by autoregressive objectives.   However, we reveal that this approach can lead to weak vision-language alignment, making these models overly dependent on costly instruction-following data.  

To address this, we introduce **EXGRA-MED**, a novel multi-graph alignment framework that jointly aligns images, instruction responses, and extended captions in the latent space, advancing semantic grounding and cross-modal coherence.   To scale to large LLMs (e.g., LLaMa-7B), we develop an efficient end-to-end training scheme using black-box gradient estimation, enabling fast and scalable optimization.  

🏆 **Key Results:**
- Matches LLAVA-MED’s performance on the Medical Visual Question Answering (VQA) task with **only 10% of pre-training data** 
- When pre-trained with **100% data** (PMC-15M), ExGra-Med (LLaMA-7B) **surpasses several SOTA Medical Multi-modal LLMs on diverse settings: (i) Medical VQA, (ii) Medical Visual Chatbot, and (iii) Zero-shot Image Classification as a VQA task**.

---

## 🚨 News

- **[Jun 2025]** 🔓 Initial codebase release (preprocessing + VQA fine-tuning).
- **[Jun 2025]** 🧩 Checkpoints for EXGRA-MED + DCI and three VQA fine-tuned models now available.
- **[Jun 2025]** 📊 Evaluation scripts and demo for data-efficiency benchmark now online.

---

## 📦 Model Checkpoints

| Model                                  | Description                                | Download Link |
|----------------------------------------|--------------------------------------------|---------------|
| `exgra-med`                            | Base EXGRA-MED model trained on 10% data   | [Link](#)     |
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


# EXGRA-MED: Extended Context Graph Alignment for Medical Vision-Language Models

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/pdf/2410.02615v3)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/MERGE-Group)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://github.com/duyhominhnguyen/Exgra-Med/blob/main/LICENSE)
[![Website](https://img.shields.io/badge/🌐%20Project%20Page-ExGra--Med-green)](https://exgra-med.github.io/)




<!--
## Collaborating Institutions

This project is a joint research effort between the following institutions:

<p align="left">
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/Stanford_logo.png" height="80" alt="Stanford University" title="Stanford University"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/eth-zurich_logo.jpg" height="120" alt="ETH Zurich" title="ETH Zurich"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/UCSD_logo.png" height="80" alt="UCSD" title="UCSD"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/Universita%CC%88t_Stuttgart_Logo.png" height="50" alt="unistuttgart" title="University of Stuttgart"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/imprs-is-logo.jpg" height="30" alt="imprs-is" title="imprs-is"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/mpi_logo.jpg" height="80" alt="mpi" title="mpi"/>
  <img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/dfki_logo.png" height="60" alt="DFKI" title="DFKI"/>
</p>

-->
---

## Abstract

State-of-the-art medical multi-modal LLMs (med-MLLMs), such as LLAVA-MED and BIOMEDGPT, primarily depend on scaling model size and data volume, with training driven largely by autoregressive objectives.   However, we reveal that this approach can lead to weak vision-language alignment, making these models overly dependent on costly instruction-following data.  

To address this, we introduce **EXGRA-MED, a novel multi-graph alignment framework that jointly aligns images, instruction responses, and extended captions** in the latent space, advancing semantic grounding and cross-modal coherence.   To scale to large LLMs (e.g., LLaMa-7B), we develop an efficient end-to-end training scheme using black-box gradient estimation, enabling fast and scalable optimization.  

<p align="center">
<img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/exgramed_v1.jpg" alt="Alt text" width="1400"/>
</p>


## 🏆 **Key Results**

:white_check_mark: **Reveals the data inefficiency of autoregressive modeling** — LLaVA-Med exhibits a significant performance drop when pre-trained on limited data, even after full fine-tuning on downstream tasks.

:white_check_mark: **Matches LLaVA-Med’s performance on Medical VQA** using only **10% of the pre-training data**, demonstrating the data efficiency of EXGRA-MED.

:white_check_mark: **Surpasses several SOTA medical multi-modal LLMs** when pre-trained on the full PMC-15M dataset (100%) with LLaMA-7B, across diverse tasks:
- (i) Medical Visual Question Answering (VQA)  
- (ii) Medical Visual Chatbot  
- (iii) Zero-shot Image Classification (as a VQA task)


<p align="center">
<img src="https://github.com/duyhominhnguyen/Exgra-Med/blob/main/assets/exgra_med_result.png" alt="Alt text" width="600"/>
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
- **[Jun 2025]** 📊 Evaluation scripts and demo for the data-efficiency benchmark for VQA are online.
- **Coming Soon** 🚧  Evaluation Scripts for Medical Visual Chatbot and Zero-shot Image Classification.
- **Coming Soon** 🚧  ExGra-Med checkpoints are trained at large-scale data with 2.5M instruction tuning samples from [MedTrinity-25M](https://arxiv.org/pdf/2408.02900) (10%).

---

## 📦 Model Checkpoints

| Model                                  | Description                                |🤗 Download Link |
|----------------------------------------|--------------------------------------------|---------------|
| `llava-med-10`                            | LLaVa-Med (10% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/llava-med-10/tree/main)     |
| `llava-med-40`                            | LLaVa-Med (40% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/llava-med-40/tree/main)     |
| `exgra-med-10`                            | ExGra-Med (10% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/exgra-med-10/tree/main)     |
| `exgra-med-40`                            | ExGra-Med (40% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/exgra-med-40/tree/main)     |
| `exgra-med`                            | Our base EXGRA-MED model (100% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/exgra-med/tree/main)     |
| `exgra-med-dci`                        | EXGRA-MED + DCI-enhanced version           | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci/tree/main)     |
| `exgra-med-dci-vqa-rad`               | Fine-tuned on VQA-RAD                      | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci-vqa-rad/tree/main)     |
| `exgra-med-dci-slake`                 | Fine-tuned on SLAKE                        | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci-slake/tree/main)     |
| `exgra-med-dci-pathvqa`               | Fine-tuned on PATH-VQA                     | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci-pathvqa/tree/main)     |

<!-- --- -->
Before starting the finetuning/inference/evaluation, download our finetuned checkpoints.
<details>
  <summary>Download Checkpoints</summary>

```bash
cd pretrained/
# pip install -U huggingface_hub
# Download MERGE-Group/llava-med-10
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/llava-med-10 --local-dir llava-med-10

# Download MERGE-Group/llava-med-40
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/llava-med-40 --local-dir llava-med-40

# Download MERGE-Group/exgra-med-10
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-10 --local-dir exgra-med-10

# Download MERGE-Group/exgra-med-40
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-40 --local-dir exgra-med-40

# Download MERGE-Group/exgra-med
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med --local-dir exgra-med

# Download MERGE-Group/exgra-med-dci
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-dci --local-dir exgra-med-dci

# Download MERGE-Group/exgra-med-dci-vqa-rad
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-dci-vqa-rad --local-dir /exgra-med-dci-vqa-rad

# Download MERGE-Group/exgra-med-dci-slake
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-dci-slake --local-dir /exgra-med-dci-slake

# Download MERGE-Group/exgra-med-dci-pathvqa
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-dci-pathvqa --local-dir /exgra-med-dci-pathvqa

```

</details>

## 🛠️ Requirements and Installation

Basic Dependencies:

- Python >= 3.10
- Pytorch
- CUDA driver

**Note**: Please check your CUDA driver to install a proper version of PyTorch. For instance, we provide a guideline for installation for CUDA 11:

```bash
conda create -n exgra-med python=3.10.12
conda activate exgra-med
pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install openai==0.27.8
pip install git+https://github.com/huggingface/transformers@cae78c46
pip install -e .
pip install einops ninja open-clip-torch shortuuid nltk
pip install --upgrade pillow
```

Also, based on your CUDA driver, please check the proper version of [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) at this [link](https://github.com/Dao-AILab/flash-attention/releases), and then install `flash-attn` package:

```bash
pip install flash-attn==2.5.7 --no-build-isolation
```

Before running the pre-training stages, please install the following graph-related packages.

```bash
pip install pyg-lib==0.3.1 \ torch-scatter==2.1.2+pt21cu118 \ torch-sparse==0.6.18+pt21cu118 \ torch-cluster==1.6.3+pt21cu118 \ torch-spline-conv==1.2.2+pt21cu118 \ -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install torch-geometric==2.4.0
pip install --no-build-isolation git+https://github.com/mrolinek/lpmp.git@9fd6211c77a14beeb68cbc3a98e1b318e614c493 
# If you meet errors during LPMP installation, verify your CMake setup and consider reinstalling it.
pip install pycocotools
```
-----
## Project Structure
* **`assets/`**: Contains various assets used by the project (e.g., images, supplementary files).
* **`scripts/`**: Houses utility bash scripts.
* **`exgra_med/`**: The main source code directory for the `exgra_med` package/application.
    * **`data_preprocessing/`**: Scripts and modules related to data preprocessing.
    * **`llava/`**: Specific modules or components related to `llava`.
        * **`eval/`**: Code for evaluating `llava` models.
        * **`instruct/`**: Code related to instructing `llava` models.
        * **`model/`**: Contains `llava` model definitions or related utilities.
        * **`notebook/`**: Jupyter notebooks for experimentation or demonstration related to `llava`.
        * **`serve/`**: Code for serving `llava` models (e.g., API endpoints).
        * **`train/`**: Training scripts and configurations for `llava`.
    * **`untar_files.py`**: A Python script possibly used for decompressing or extracting files.
* **`LICENSE`**: The license under which the project is distributed.
* **`pyproject.toml`**: A file used for specifying project build system requirements and project metadata (part of PEP 517/518).
* **`README.md`**: This README file, providing an overview of the project.

--------
## 📄 Dataset Configuration Files

 🔗 **Downstream Stage**:
We provide pre-built `.json` configuration files for all datasets used in VQA training and evaluation in downstream tasks. These files specify paths, splits, and preprocessing parameters necessary for seamless execution. Firstly, create each dataset folder in folder `data/`, then put the corresponding dataset `.json` files into folders. Next, please see websites for datasets [VQA-RAD](https://www.kaggle.com/datasets/shashankshekhar1205/vqa-rad-visual-question-answering-radiology), [SLAKE 1.0](https://www.med-vqa.com/slake/), and [PATH-VQA](https://github.com/KaveeshaSilva/PathVQA) to download `images/` folders and upload them into corresponding dataset folders in `data/` folder.

| Dataset      | Task       | Config File Description       | Download Link              |
| ------------ | ---------- | ----------------------------- | -------------------------- |
| VQA-RAD      | VQA        | Train/val splits, QA pairs    | [link](https://huggingface.co/datasets/MERGE-Group/VQA-RAD) |
| SLAKE        | VQA        | Train/val splits, QA pairs    | [link](https://huggingface.co/datasets/MERGE-Group/SLAKE)    |
| PATH-VQA     | VQA        | Train/val splits, QA pairs    | [link](https://huggingface.co/datasets/MERGE-Group/PATH-VQA)  |

 🔗 **Pre-Training Stage**:
To prepare dataset for pre-training stage using both **Exgra-Med** and original **LLaVA-Med** algorithms, downloading `.json` files for stage 1 (alignment) and stage 2 (instruction) in this [link](https://huggingface.co/datasets/MERGE-Group/Extended-Caption-GPT). Next, create new folder `pretraining_data/` in `data/` folder and upload downloaded jsons combined with `images/` folder into `pretraining_data/` folder. Please note that update dataset file paths inside `.sh` training files in `scripts/` folder if needed to match your local dataset locations.


-----
## 🖇️ Extended Instructions Generation
The script [`extended_caption_generation.py`](exgra_med/data_preprocessing/extended_caption_generation.py) reads an input JSON of instructions/conversations, sends each question+answer pair to an LLM with a provided [system prompt](exgra_med/prompts/extend_caption.txt), and replaces the answer with the LLM-provided revision. It supports resuming from an existing extended output file.

Input JSON should contain items with a `conversations` (or misspelled `conversatons`) key whose value is a list of role objects. The script pairs even-indexed entries (questions) with the following odd-indexed entries (answers) and updates the answer `value` with the LLM `revision`.

Output file: if not resuming, a timestamped file is created next to the input file with suffix `_extended_<model>_<timestamp>.json`. When resuming, pass `--resume_from` to continue.

Create a `.env` file (or set environment variables) with the OpenRouter/OpenAI endpoint and API key used by the `OpenAI` client. Example `.env`:

```plaintext
OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-...
```

Basic usage:

```python
python extended_caption_generation.py
  --original_instruction_fpath path/to/original_instructions.json
  --system_prompt_fpath path/to/system_prompt.txt
```

Options:
- `--original_instruction_fpath` (required): Path to input JSON with instructions/conversations.
- `--system_prompt_fpath` (required): Path to a text file containing the system prompt to send to the LLM.
- `--resume_from` (optional): Path to an existing extended JSON to resume from (skips already-processed ids).
- `--model_name` (optional): LLM model identifier (default: `openai/gpt-4o-mini`). List of reported models: openai/gpt-4o, google/gemini-2.5-flash, qwen/qwen3-8b.

Example with model and resume:

```python
python extended_caption_generation.py
  --original_instruction_fpath data/original.json
  --system_prompt_fpath prompts/system_prompt.txt
  --model_name openai/gpt-4o-mini
  --resume_from data/original_extended_gpt-4o-mini_20250101_120000.json
```

## Pre-Training on Two Stages

We provide ready-to-use scripts to pre-train **EXGRA-MED** on 2 stages named `stage1.sh` and `stage2.sh` in `scripts/` folder after downloading pre-training dataset. Make sure to update the `--model_name_or_path`, `--data_path`, and `--image_folder` in each `.sh` file to point to the correct location of the downloaded model and dataset.

```bash
# Example: Pre-train EXGRA-Med on stage 1
bash scripts/stage1.sh

# Example: Pre-train EXGRA-Med on stage 2
bash scripts/stage2.sh
```

## 🔧 Fine-tuning on VQA Tasks
We provide ready-to-use scripts to fine-tune **EXGRA-MED** and **EXGRA-MED + DCI** on three popular medical VQA benchmarks: **VQA-RAD**, **SLAKE**, and **PATH-VQA**.

Each script uses one of our pretrained checkpoints as the starting point.  👉 **Before running**, make sure to update the `--model_name_or_path` in each `.sh` file to point to the correct location of the downloaded model.

```bash
# Example: Fine-tune on VQA-RAD
bash scripts/llava1-5_stage2_data_rad.sh         # without DCI
bash scripts/llava1-5_stage2_data_rad_dci.sh     # with DCI

# Fine-tune on SLAKE
bash scripts/llava1-5_stage2_slake.sh            # without DCI
bash scripts/llava1-5_stage2_slake_dci.sh        # with DCI

# Fine-tune on PATH-VQA
bash scripts/llava1-5_stage2_pvqa.sh            # without DCI
bash scripts/llava1-5_stage2_pvqa_dci.sh        # with DCI
```

-----
## 📈 Evaluation

You can run evaluation for each of the three key tasks:

## 1. Medical VQA Evaluation

```bash
# supports VQA-RAD, SLAKE, PATH-VQA

# change the following
# --model-name: Path to load the model from finetuning stage
# --answers-file: file to store the result (i.e the answers to the medical question)
python exgra_med/llava/eval/run_med_datasets_eval_batch.py \
--num-chunks 2 \
--model-name \<output_vqa_rad_checkpoint\> \
--mm_dense_connector_type none \
--num_l 6 \
--question-file ./data_RAD/test_w_options_new.json \
--image-folder ./data_RAD/images \
--answers-file \<answers_file\>

#change the following
#--pred: same as --answers-file above
# the metrics (recall and accuracy) are saved as a text file in the same place, with the same name as --pred. 
#E.g: if --pred is ans-opt-new-3.jsonl, then metrics are saved in ans-opt-new-3.txt
python exgra_med/llava/eval/run_eval.py \
--gt ./data_RAD/test_w_options_new.json \
--pred \<answers_file\> \
--candidate ./data_RAD/candidate.json
```

## 2. Medical Visual Chatbot

🚧 **To be updated!**
```
bash scripts/eval_chatbot.sh
```

## 3. Zero-shot Image Classification
By reformulating image classification as visual question answering, we can generate predictions by solving the VQA task with multiple-choice questions. First, download OmniMedVQA benchmark  from [OpenDataLab](https://openxlab.org.cn/datasets/GMAI/OmniMedVQA) or [huggingface](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA/tree/main) and unzip it. Follow instructions in the file [`zero_shot_classification.sh`](scripts/zero_shot_classification.sh) to change variables `FILENAME`, `OUTPUTNAME`, `WORKDIR` and `MODEL_CKPT` and `CONNECTOR_TYPE`. Then run:

```
bash scripts/zero_shot_classification.sh
```

------
## 🔬 Data Efficiency Demonstration (10% vs 40%)
To replicate our findings on LLAVA-MED’s data inefficiency and the strength of EXGRA-MED with 10% and 40% data (Tables 1 & 2 in the paper):


```bash
# Fine-tune EXGRA-MED with 10%/40% data on VQA task
bash scripts/train_exgra_10percent.sh #
```

```bash
# Fine-tune checkpoint LLaVa-Med with 10%/40% data on VQA task
bash scripts/train_llava_10percent.sh
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


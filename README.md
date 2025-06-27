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
- **[Jun 2025]** 📊 Evaluation scripts and demo for data-efficiency benchmark now online.

---

## 📦 Model Checkpoints

| Model                                  | Description                                |🤗 Download Link |
|----------------------------------------|--------------------------------------------|---------------|
| `llava-med-10`                            | LLaVa-Med (10% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/llava-med-10)     |
| `llava-med-40`                            | LLaVa-Med (40% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `exgra-med-10`                            | ExGra-Med (10% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/exgra-med-10)     |
| `exgra-med-40`                            | ExGra-Med (40% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/exgra-med-40)     |
| `exgra-med`                            | Our base EXGRA-MED model (100% pre-trained PMC-15M)                   | [Link](https://huggingface.co/MERGE-Group/exgra-med)     |
| `exgra-med-dci`                        | EXGRA-MED + DCI-enhanced version           | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci)     |
| `exgra-med-dci-vqa-rad`               | Fine-tuned on VQA-RAD                      | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci-vqa-rad)     |
| `exgra-med-dci-slake`                 | Fine-tuned on SLAKE                        | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci-slake)     |
| `exgra-med-dci-pathvqa`               | Fine-tuned on PATH-VQA                     | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci-pathvqa)     |

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
conda create -n exgra-med python=3.10
conda activate exgra-med
pip install --upgrade pip
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install openai==0.27.8
pip install git+https://github.com/huggingface/transformers@cae78c46
pip install -e .
pip install einops ninja open-clip-torch
```

Also, based on your CUDA driver, please check the proper version of [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) at this [link](https://github.com/Dao-AILab/flash-attention/releases), and then install `flash-attn` package:

```bash
pip install flash-attn --no-build-isolation
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
We provide pre-built `.json` configuration files for all datasets used in VQA training and evaluation. These files specify paths, splits, and preprocessing parameters necessary for seamless execution.

| Dataset      | Task       | Config File Description       | Download Link              |
| ------------ | ---------- | ----------------------------- | -------------------------- |
| VQA-RAD      | VQA        | Train/val splits, QA pairs    | [vqa\_rad\_config.json](#) |
| SLAKE        | VQA        | Train/val splits, QA pairs    | [slake\_config.json](#)    |
| PATH-VQA     | VQA        | Train/val splits, QA pairs    | [pathvqa\_config.json](#)  |

To download our language-image multimodal instruction-folllowing dataset, please run the following script:

```
bash scripts/download_data.sh
```


 🔗 **Instructions**:

- Place the downloaded `.json` files under the `configs/datasets/` directory.

- Update file paths inside if needed to match your local dataset locations.

-----
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


```bash
# Fine-tune EXGRA-MED with 10%/40% data on VQA task
bash scripts/train_exgra_10percent.sh
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


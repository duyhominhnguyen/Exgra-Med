#!/bin/bash
export WORKDIR=$(pwd)/exgra_med
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

EXP=stage1
DATE=_100_scale
# contrastive loss type: infonce, siglip, directOT (none contrastive loss type)
# vision_tower: openai/clip-vit-large-patch14 microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25003 exgra_med/llava/train/train_mem_pre.py \
        --model_name_or_path ${DATADIR}/weights/LLaVA-7b-v0 \
        --data_path ${DATADIR}/data/alignment/llava_med_alignment_100_2_conversations.json \
        --image_folder ${DATADIR}/data/images \
        --vision_tower openai/clip-vit-large-patch14 \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --mm_dense_connector_type none \
        --contrastive False \
        --after_de False \
        --multi_graph False \
        --alpha 1.0 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end \
        --bf16 True \
        --output_dir ${WORKDIR}/models/checkpoint_llava-med-7b-pretrain_version_1-5${DATE} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2400 \
        --save_total_limit 1 \
        --learning_rate 2e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name ${EXP}_${DATE}


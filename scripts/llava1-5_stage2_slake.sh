#!/bin/bash
export WORKDIR=$(pwd)/exgra_med
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
lr=4e-5
version=_exgra_med_100_scale_${lr}
model_name_or_path=./models/checkpoint_llava_med_instruct_60k_inline_mention_version_1-5${version}
output_dir=./weights_finetuned/slake-100${version}
run_name=slake-100${version}


torchrun --nnodes=1 --nproc_per_node=2 --master_port=25057 \
    llava/train/train_mem.py \
    --model_name_or_path=${model_name_or_path} \
    --data_path ./Slake1.0/train_w_options_new.json \
    --image_folder ./Slake1.0/images \
    --evaluate_on_val \
    --evaluate_on_test \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_dense_connector_type none \
    --num_l 6 \
    --bf16 True \
    --output_dir=${output_dir} \
    --num_train_epochs 15 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 116 \
    --load_best_model_at_end \
    --metric_for_best_model "eval_val_average" \
    --save_strategy "steps" \
    --save_steps 11600 \
    --save_total_limit 1 \
    --learning_rate=${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name=${run_name}
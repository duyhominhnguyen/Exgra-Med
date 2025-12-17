if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

EXP=stage2
DATE=_1e0_multi_graph_40_scale
# contrastive loss type: infonce, siglip, directOT (none contrastive loss type)
# vision_tower: openai/clip-vit-large-patch14 microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
STAGE1_CKPT=./models/checkpoint_llava-med-7b-pretrain_version_1-5_40_scale

srun -p A100-80GB,A100-IML -t 1-10:59:59 --ntasks 1 \
        --gpus-per-task 4 \
        --cpus-per-gpu=6 \
        --mem-per-cpu 40G \
        --container-image=${ROOTDIR}/Research/Nghiem_LLaVA-Med/exgra_med_finetune.sqsh \
        --container-workdir="`pwd`" \
        --container-mounts=${WORKDIR}:/root/LLaVA-Med,${ROOTDIR}:${ROOTDIR},/ds:/ds:ro,"`pwd`":"`pwd`" \
        --export="NCCL_IB_DISABLE=1" \
        --export="OMP_NUM_THREADS=10" \
        --export="LOGLEVEL=INFO" \
        --export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
        --export="FI_PROVIDER='efa'" \
        --export="CUDA_LAUNCH_BLOCKING=1" \
        --export="CUDA_VISIBLE_DEVICES=0,1,2,3" \
        --export="WANDB_API_KEY=${WANDB_API_KEY}" \
        torchrun --nnodes=1 --nproc_per_node=4 --master_port=25010 llava/train/train_mem_pre.py \
        --model_name_or_path ${STAGE1_CKPT} \
        --data_path ${DATADIR}/data/instruct/llava_med_instruct_40_inline_mention_2_conversations.json \
        --image_folder ${DATADIR}/data/images \
        --vision_tower openai/clip-vit-large-patch14 \
        --mm_projector_type mlp2x_gelu \
        --mm_dense_connector_type none \
        --contrastive True \
        --after_de True \
        --multi_graph True \
        --alpha 1.0 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end True \
        --fp16 True \
        --output_dir ${WORKDIR}/models/checkpoint_llava_med_instruct_60k_inline_mention_version_1-5${DATE} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 5000 \
        --save_total_limit 3 \
        --learning_rate 2e-5 \
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
        --run_name ${EXP}_${DATE}


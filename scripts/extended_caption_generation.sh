#!/bin/bash
export WORKDIR=$(pwd)/exgra_med
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
## List of used models: openai/gpt-4o-mini, google/gemini-2.5-flash, qwen/qwen3-8b
python exgra_med/data_preprocessing/extended_caption_generation.py \
    --original_instruction_fpath ./data/llava_med_instruct_remain_inline_mention.json \
    --system_prompt_fpath ./exgra_med/prompts/extend_caption.txt \
    --model_name qwen/qwen3-8b
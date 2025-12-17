# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import wandb
import os

from llava.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

from llava.train.train_pre import train


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "llava_med"

replace_llama_attn_with_flash_attn()


if __name__ == "__main__":

    wandb.login(key=WANDB_API_KEY)
    train()

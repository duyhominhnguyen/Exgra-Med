from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field
import random
import copy
import os
import torch
import pathlib
import logging
import math
# import h5py
import numpy as np
from PIL import Image
import transformers
from llava.train.train_original_pre import (
    DataArguments,
    TrainingArguments,
    ModelArguments,
    SupervisedDataset,
    LazySupervisedDataset,
    DataCollatorForSupervisedDataset,
    preprocess_v1,
    _tokenize_fn,
    _add_speaker_and_signal,
    _mask_targets,
    preprocess_multimodal,
    smart_tokenizer_and_embedding_resize,
    patch_FSDP_use_orig_params,
    safe_save_model_for_hf_trainer,
    IGNORE_INDEX,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN
)

from llava.model.llava_pre import LlavaLlamaForCausalLM
from llava import conversation as conversation_lib
from llava.train.llava_trainer import LLaVATrainer

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
import inspect
import datetime
from llava.utils import build_logger

logger = logging.getLogger("LVLM-Med")


@dataclass
class LVLMModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_dense_connector_type: Optional[str] = field(default="none")
    contrastive: bool = field(default=False)
    alpha: float = field(default=1.0)
    after_de: bool = field(default=False)
   
    ## Multi-Graph
    multi_graph: bool = field(default=False)
    graph_num_features: int = field(default=4096)

def preprocess_multimodal_combine_long(
    sources: Sequence[str],
    sources_long: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    """
    This function is used to preprocess sentences for both 
    original and extended version of questions and answers.
    """
    is_multimodal = multimodal_cfg["is_multimodal"]
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    if not is_multimodal:
        return sources, sources_long

    ########## Prepare the orginal version of QAs ##########
    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg["use_im_start_end"]:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )

            if isinstance(sentence["value"], int):
                sentence["value"] = str(sentence["value"])
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    ########## Prepare the extended version of QAs ##########
    for source in sources_long:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg["use_im_start_end"]:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )

            if isinstance(sentence["value"], int):
                sentence["value"] = str(sentence["value"])
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources, sources_long

def merge_sentences_preprocess(
    sources: Sequence[str], sources_long: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, sep: str = ". "
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    5. Moreover, this function also prepare language samples for multi-graph alignment.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer), preprocess_v1(sources_long, tokenize)
    # add end signal and concatenate together

    num_step = 2
    i_step = 1

    ########## Prepare the original version of answers for multi-graph alignment ##########
    list_source_sentences = []
    for source in sources:
        list_merge_sentences = []
        for i in range(0, len(source), num_step):
            sent = source[i + i_step]["value"]
            sent = sent.replace("\n", "")
            sent = sent.replace("<im_patch>", "")
            sent = sent.replace("<im_end>", "")
            sent = sent.replace("<im_start>", "")
            list_merge_sentences.append(sent)
        merge_sentence = sep.join(list_merge_sentences)
        list_source_sentences.append(merge_sentence)

    merge_sentence_id = _tokenize_fn(list_source_sentences, tokenizer)["input_ids"]

    ########## Prepare the extended version of answers for multi-graph alignment ##########
    list_source_sentences_long = []
    for source in sources_long:
        list_merge_sentences_long = []
        for i in range(0, len(source), num_step):
            sent = source[i + i_step]["value"]
            sent = sent.replace("\n", "")
            sent = sent.replace("<im_patch>", "")
            sent = sent.replace("<im_end>", "")
            sent = sent.replace("<im_start>", "")
            list_merge_sentences_long.append(sent)
        merge_sentence_long = sep.join(list_merge_sentences_long)
        list_source_sentences_long.append(merge_sentence_long)

    merge_sentence_id_long = _tokenize_fn(list_source_sentences_long, tokenizer)["input_ids"]
    
    ########## Prepare the answers for autoregressive ##########
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"

        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn(
            [header] + [s["value"] for s in source], tokenizer
        )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(
        input_ids=input_ids, labels=targets, merge_sentence_id=merge_sentence_id, merge_sentence_id_long=merge_sentence_id_long
    )

class LVLMLazySupervisedDataset(LazySupervisedDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
    ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, multimodal_cfg=multimodal_cfg)
        if ".h5" in self.multimodal_cfg["image_folder"]:
            self.h5f = h5py.File(self.multimodal_cfg["image_folder"], 'r')

    def __del__(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()
        # self._shutdown_workers()

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.multimodal_cfg["image_folder"]
            processor = self.multimodal_cfg["image_processor"]
            try:
                if ".h5" in self.multimodal_cfg["image_folder"]:
                    image = np.array(self.h5f[image_file])
                else:
                    image = Image.open(os.path.join(image_folder, image_file)).convert(
                        "RGB"
                    )
            except Exception as exn:
                logger.error(exn)

                return random.choice(self)

            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.multimodal_cfg["image_aspect_ratio"] == "keep":
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(
                    image,
                    return_tensors="pt",
                    do_center_crop=False,
                    size={"shortest_edge": shortest_edge},
                )["pixel_values"][0]
            elif self.multimodal_cfg["image_aspect_ratio"] == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]

            # import pdb; pdb.set_trace()
            image_token_len = self.multimodal_cfg["image_token_len"]
            patch_size = int(image.shape[1] // math.sqrt(image_token_len))
            cur_token_len = (image.shape[1] // patch_size) * (
                image.shape[2] // patch_size
            )  # FIXME: 14 is hardcoded patch size

            try:
                sources_long = copy.deepcopy([e["conversations_long"] for e in sources])
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources_long = copy.deepcopy([e["conversatons_long"] for e in sources])
                sources = copy.deepcopy([e["conversatons"] for e in sources])

            sources, sources_long = preprocess_multimodal_combine_long(sources, sources_long, self.multimodal_cfg, cur_token_len)
        else:
            try:
                sources_long = copy.deepcopy([e["conversations_long"] for e in sources])
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources_long = copy.deepcopy([e["conversatons_long"] for e in sources])
                sources = copy.deepcopy([e["conversatons"] for e in sources])

        data_dict = merge_sentences_preprocess(
            sources=sources, sources_long=sources_long, tokenizer=self.tokenizer, sep=". "
        )
 
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                merge_sentence_id=data_dict["merge_sentence_id"][0],
                merge_sentence_id_long=data_dict["merge_sentence_id_long"][0],
            )
        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.multimodal_cfg["is_multimodal"]:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg["image_processor"].crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        return data_dict


class LVLMDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, merge_sentence_id, merge_sentence_id_long = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "merge_sentence_id", "merge_sentence_id_long")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        merge_sentence_id = torch.nn.utils.rnn.pad_sequence(
            merge_sentence_id,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        merge_sentence_id_long = torch.nn.utils.rnn.pad_sequence(
            merge_sentence_id_long,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            merge_sentence_id=merge_sentence_id,
            merge_sentence_id_long=merge_sentence_id_long,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            attention_mask_F=merge_sentence_id.ne(self.tokenizer.pad_token_id),
            attention_mask_F_long=merge_sentence_id_long.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LVLMLazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        multimodal_cfg=dict(
            is_multimodal=data_args.is_multimodal,
            image_token_len=data_args.image_token_len,
            image_folder=data_args.image_folder,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=getattr(data_args, "mm_use_im_start_end", False),
            image_processor=getattr(data_args, "image_processor", None)
        )
    )
    data_collator = LVLMDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    parser = transformers.HfArgumentParser(
        (LVLMModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    run_id = datetime.datetime.now()
    run_id = str(run_id).replace(" ", "-")
    run_id = run_id.split(".")[0]
    os.makedirs(training_args.output_dir, exist_ok=True)
    build_logger(
        logger_name="LVLM-Med",
        logger_filename=os.path.join(training_args.output_dir, "log.txt"),
    )
    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1_1"
        ]

    ## TODO: increase gradient_accumulation_steps N times compare to decrease number of used GPUs
    model_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps
   
    if model_args.vision_tower is not None:
        model_vision_dict = model.model.initialize_vision_modules(
            model_args=model_args,
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        )
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
      
        model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
        vision_config = model_vision_dict["vision_config"]

        data_args.image_token_len = model_vision_dict["image_token_len"]
        data_args.image_processor = model_vision_dict["image_processor"]
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )

        if model_args.tune_mm_mlp_adapter:

            model.requires_grad_(False)
            for p in model.model.mm_projector.parameters():
                p.requires_grad = True
            
            if model_args.contrastive and model_args.multi_graph:
                for p in model.model.message_pass_node_features.parameters():
                    p.requires_grad = True
                    

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.model.mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        vision_config.use_im_start_end = training_args.use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.initialize_vision_tokenizer(
            mm_use_im_start_end=model_args.mm_use_im_start_end,
            tokenizer=tokenizer,
            device=training_args.device,
            tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        )

        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    logger.warning(
                        "Attempting to use FSDP while {} parameters do not require gradients: {}".format(
                            len(params_no_grad), params_no_grad
                        )
                    )
                else:
                    logger.warning(
                        "Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)".format(
                            len(params_no_grad), ", ".join(params_no_grad[:10])
                        )
                    )
                logger.warning(
                    "Attempting to use FSDP with partially frozen paramters, this is experimental."
                )
                logger.warning(
                    "As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
                )
                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

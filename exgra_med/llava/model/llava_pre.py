from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    CLIPVisionModel,
    CLIPImageProcessor,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from llava.model.utils import *
import open_clip
import os, json
import re

import torch_geometric
from torch_geometric import loader
from torch_geometric.data import Data
from einops import rearrange, reduce, repeat
from torch_geometric.nn import SplineConv, EGConv, GraphConv, GATConv, GCNConv

from lpmp_py import GraphMatchingModule
from llava.model.dense_connector import dense_connector

# from torch_sinkhorn.problem import Epsilon, LinearProblem
# from torch_sinkhorn.sinkhorn import Sinkhorn
import logging
from llava.model.llava_original_pre import LlavaLlamaModel as LlavaLlamaModelBase
from llava.model.llava_original_pre import (
    LlavaLlamaForCausalLM as LlavaLlamaForCausalLMBase,
)

from llava.utils_graph import utils
from llava.utils_graph.datasets import build_loader
from llava.utils_graph.distributed import init_distributed_mode
from llava.utils_graph.build_graph import (
    build_graphs_main_1,
    calculate_edge_attr_1,
)
from llava.utils_graph import config as cfg

from llava.utils_graph.utils import (
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    lexico_iter,
)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

LOG_DIR = "outputs/log"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("LVLM-Med")


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    logger.info(f"projector_type: {projector_type}")
    if projector_type == "linear":
        logger.info(
            "--------------------------This is version 1.0---------------------"
        )
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        logger.info(
            "--------------------------This is version 1.5---------------------"
        )
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-4
    for i in range(500):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(1, 0).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T


def distributed_sinkhorn(out, vocab_dist):
    """
    Optimal Transport for VLAP
    """
    Q = torch.exp(out / 0.1).t()
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]

    for it in range(500):
        Q *= (vocab_dist / 2.5e-1).softmax(dim=0)
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment Q: K x B
    return Q.t()  # B x K or B*Ns x K


def average_similarity(
    align_image_features: torch.Tensor,
    image_features: torch.Tensor,
    align_input_embeds: torch.Tensor,
    temperature: torch.Tensor,
):
    matrixSimi = torch.zeros(
        align_image_features.shape[0],
        align_image_features.shape[0],
        device=image_features.device,
        dtype=image_features.dtype,
    )
    align_image_features_pool = align_image_features.mean(dim=1)

    align_image_features_pool = (
        align_image_features_pool / align_image_features_pool.norm(dim=1)[:, None]
    )

    for i, align_input in enumerate(align_input_embeds):
        align_input_norm = align_input / align_input.norm(dim=1)[:, None]
        simi = torch.mm(align_image_features_pool, align_input_norm.transpose(0, 1))

        simi = simi.mean(dim=1)
        matrixSimi[:, i] = torch.exp(temperature) * simi
    return matrixSimi


def optimal_transport_similarity(
    align_image_features: torch.Tensor,
    image_features: torch.Tensor,
    align_input_embeds: torch.Tensor,
    temperature: torch.Tensor,
):
    matrixSimi = torch.zeros(
        align_image_features.shape[0],
        align_image_features.shape[0],
        device=image_features.device,
        dtype=image_features.dtype,
    )
    for i, image_feat in enumerate(align_image_features):
        image_feat_norm = image_feat / image_feat.norm(dim=1)[:, None]
        for j, align_input in enumerate(align_input_embeds):
            u = (
                torch.ones(
                    image_feat.shape[0],
                    device=image_features.device,
                    dtype=image_features.dtype,
                )
                / image_feat.shape[0]
            )
            v = (
                torch.ones(
                    align_input.shape[0],
                    device=image_features.device,
                    dtype=image_features.dtype,
                )
                / align_input.shape[0]
            )

            align_input_norm = align_input / align_input.norm(dim=1)[:, None]
            simi = torch.mm(image_feat_norm, align_input_norm.transpose(0, 1))
            cost = 1 - simi
            with torch.no_grad():
                KK = torch.exp(-cost / 0.1)
                T = Sinkhorn(KK, u, v)
            if torch.isnan(T).any():
                return None

            sim_op = torch.sum(T * simi)
            matrixSimi[i, j] = (sim_op) * torch.exp(temperature)
    return matrixSimi


def compute_loss(
    contrastive_loss_type: str,
    simi_type: str,
    align_image_features: torch.Tensor,
    image_features: torch.Tensor,
    matrixSimi: torch.Tensor,
    beta: torch.Tensor,
):
    if contrastive_loss_type == "infonce" and simi_type != "directOT":
        targetAlign = torch.arange(align_image_features.shape[0]).to(
            image_features.device
        )
        lossAlign = CrossEntropyLoss()(matrixSimi, targetAlign)
    elif contrastive_loss_type == "siglip" and simi_type != "directOT":
        matrixSimi += beta
        targetAlign = (
            2 * torch.eye(align_image_features.shape[0])
            - torch.ones(align_image_features.shape[0])
        ).to(image_features.device)
        lossAlign = (
            -1.0
            * torch.sum(torch.nn.LogSigmoid()(matrixSimi * targetAlign))
            / align_image_features.shape[0]
        )
    elif contrastive_loss_type == "none" and simi_type == "directOT":
        lossAlign = lossAlign / align_image_features.shape[0]
    else:
        raise ValueError("Wrong contrastive loss type.")

    return lossAlign


class GDPModel1(torch.nn.Module):
    """
    Applying GNN on built graphs.
    """

    def __init__(self, aggr="max", in_features=512, out_features=512):
        super().__init__()

        self.conv1 = GraphConv(
            in_channels=in_features, out_channels=out_features, aggr=aggr
        )
        self.conv2 = GraphConv(
            in_channels=out_features, out_channels=out_features, aggr=aggr
        )

    def forward(self, data):
        typeData = data.type
        x_old, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        ini_size_x = len(x_old) // len(typeData)
        ini_size_edge = edge_index.shape[1] // len(typeData)
        # logger.info(x_old.shape)

        x = self.conv1(x_old, edge_index)

        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        data.x = x
        return data


class SiameseNodeFeaturesToEdgeFeatures(torch.nn.Module):
    """
    Calculate the attributes of edges through corresponding nodes.
    """

    def __init__(self):
        super(SiameseNodeFeaturesToEdgeFeatures, self).__init__()

    #         self.num_edge_features = total_num_nodes
    def forward(self, graph):
        orig_graphs = graph.to_data_list()
        orig_graphs = [self.vertex_attr_to_edge_attr(graph) for graph in orig_graphs]
        return orig_graphs

    def vertex_attr_to_edge_attr(self, graph):
        """Assigns the difference of node features to each edge"""
        flat_edges = graph.edge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)

        new_shape = (graph.edge_index.shape[1], 2, vertex_attrs.shape[1])
        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        new_edge_attrs = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        graph.edge_attr = new_edge_attrs
        return graph


class InnerProductWithWeightsAffinity(nn.Module):
    """
    Calculate similarity score between two embedding vectors.
    """

    def __init__(self, cosine_similarity, activation):
        self.cosine_similarity = cosine_similarity
        self.activation = activation
        super(InnerProductWithWeightsAffinity, self).__init__()

    def forward(self, a, b):
        res = None
        if self.cosine_similarity:
            eps = 1e-8
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.clamp(a_n, min=eps)
            b_norm = b / torch.clamp(b_n, min=eps)
            res = torch.matmul(a_norm, b_norm.transpose(0, 1))
        else:
            res = torch.matmul(a, b.transpose(0, 1))

        if self.activation == "softplus":
            res = torch.nn.functional.softplus(res)
        elif self.activation == "relu":
            res = torch.nn.functional.softplus(res)
        return res


class HammingLoss(torch.nn.Module):
    """
    This class is Hamming loss for multi-graph alignment.
    """

    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


class LlavaLlamaModel(LlavaLlamaModelBase):
    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super().__init__(config, mm_vision_tower, mm_hidden_size)
        if hasattr(config, "mm_projector_type"):
            self.mm_projector = build_vision_projector(config)

    def initialize_vision_modules(
        self,
        model_args,
        vision_tower,
        mm_vision_select_layer,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
    ):

        self.contrastive = getattr(model_args, "contrastive", False)
        self.grad_step = model_args.gradient_accumulation_steps
        self.mm_dense_connector_type = model_args.mm_dense_connector_type
        if self.contrastive:

            self.alpha = getattr(model_args, "alpha", 1.0)
            self.multi_graph = model_args.multi_graph
            self.config.multi_graph = model_args.multi_graph

            if (
                not hasattr(self, "build_edge_features_from_node_features")
                and self.multi_graph
            ):
                logger.info("Initial Multi-Graph")
                self.build_edge_features_from_node_features = (
                    SiameseNodeFeaturesToEdgeFeatures()
                )
                self.vertex_affinity = InnerProductWithWeightsAffinity(
                    cosine_similarity=True, activation=None
                )
                self.edge_affinity = InnerProductWithWeightsAffinity(
                    cosine_similarity=False, activation=None
                )
                logger.info("Do not remove graph")
                self.message_pass_node_features = GDPModel1(
                    aggr="max",
                    in_features=model_args.graph_num_features,
                    out_features=model_args.graph_num_features,
                )

                self.hamming_loss = HammingLoss()
                self.config.graph_num_features = model_args.graph_num_features
            else:
                logger.info("Multi-Graph already exists or you do not use multi-graph")

            self.after_de = model_args.after_de

            if self.multi_graph:
                logger.info("Using Multi-Graph")

        if "BiomedCLIP" in vision_tower:
            self.vision_tower_name = (
                "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            return self.initialize_vision_modules_from_biomed_clip(
                model_args,
                vision_tower,
                mm_vision_select_layer,
                pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
                tune_mm_mlp_adapter=False,
            )
        else:
            return self.initialize_vision_modules_from_openai_clip(
                model_args,
                vision_tower,
                mm_vision_select_layer,
                pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
                tune_mm_mlp_adapter=False,
            )

    def initialize_vision_modules_from_openai_clip(
        self,
        model_args,
        vision_tower,
        mm_vision_select_layer,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
    ):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, "vision_tower"):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )

        if self.mm_dense_connector_type in ["dci", "sci"]:
            self.config.mm_hidden_size = vision_config.hidden_size * 3
        else:
            self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, "mm_projector"):
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )
            self.mm_projector.load_state_dict(
                {k.split(".")[-1]: v for k, v in mm_projector_weights.items()}
            )

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config,
        )

    def initialize_vision_modules_from_biomed_clip(
        self,
        model_args,
        vision_tower,
        mm_vision_select_layer,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
    ):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        openai_vision_tower = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        vision_config = openai_vision_tower.config
        del openai_vision_tower

        if not hasattr(self, "vision_tower"):
            model, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            vision_tower = (
                model.visual.trunk
            )  # Please refer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py#LL60C18-L60C18

            # from huggingface_hub import snapshot_download
            # BiomedCLIP_file_path = "biomed-clip-share"
            # # snapshot_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", local_dir=BiomedCLIP_file_path)
            # with open(os.path.join(BiomedCLIP_file_path, "open_clip_config.json"), 'r') as file:
            #     config = json.load(file)

        else:
            vision_tower = self.vision_tower[0]

        setattr(vision_tower, "config", vision_config)
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )

        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, "mm_projector"):
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )
            self.mm_projector.load_state_dict(
                {k.split(".")[-1]: v for k, v in mm_projector_weights.items()}
            )

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config,
        )

    def extract_visual_features(self, vision_tower, images):
        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)

        if (
            "BiomedCLIP" in self.vision_tower_name
            or "biomed_clip" in self.vision_tower_name
        ):
            image_forward_outs = vision_tower.get_intermediate_layers(
                images, n=3
            )  # take last n blocks if n is an int, if in is a sequence, select by matching indices
            image_features = image_forward_outs[select_hidden_state_layer]
            image_features = image_features
            dummy_image_features = torch.zeros(
                196, 768, device=image_features.device, dtype=image_features.dtype
            )
            if self.mm_dense_connector_type in ["sti", "sci", "dci"]:
                image_features = dense_connector(
                    image_features=image_features,
                    image_forward_outs=image_forward_outs,
                    mm_dense_connector_type=self.mm_dense_connector_type,
                )
        else:
            image_forward_outs = vision_tower(images, output_hidden_states=True)
            select_hidden_state = image_forward_outs.hidden_states[
                select_hidden_state_layer
            ]
            image_features = select_hidden_state[:, 1:]
            if self.mm_dense_connector_type in ["dci", "sci"]:
                dummy_image_features = torch.zeros(
                    256, 3072, device=image_features.device, dtype=image_features.dtype
                )
            else:
                dummy_image_features = torch.zeros(
                    256, 1024, device=image_features.device, dtype=image_features.dtype
                )
            if self.mm_dense_connector_type in ["sti", "sci", "dci"]:
                image_features = dense_connector(
                    image_features=image_features,
                    image_forward_outs=image_forward_outs,
                    mm_dense_connector_type=self.mm_dense_connector_type,
                )
        return image_features, dummy_image_features

    def build_graph(self, global_map1, global_map2):
        """
        This function is used to build graph for two embedding inputs
        (use structure of graph 1 to create graph 2 with similar structure).
        """
        map_len = global_map1[0].shape[0]
        edge_indice1, edge_feature1 = build_graphs_main_1(
            global_map1[0], n_neighbors=5, num_workers=1
        )
        # duplicate graph structure
        edge_indice2 = edge_indice1
        # create graph feature from given structure
        edge_feature2 = calculate_edge_attr_1(
            edge_indice2, global_map2[0], n_neighbors=5, num_workers=1
        )

        data_graph1 = Data(
            x=global_map1[0],
            edge_index=edge_indice1,
            edge_attr=edge_feature1,
            type=global_map1[1],
        )
        data_graph2 = Data(
            x=global_map2[0],
            edge_index=edge_indice2,
            edge_attr=edge_feature2,
            type=global_map2[1],
        )
        graph_loader = loader.DataLoader([data_graph1, data_graph2], batch_size=2)

        graph = self.message_pass_node_features(next(iter(graph_loader)))

        orig_graph = self.build_edge_features_from_node_features(graph)

        # add norm for GNN
        for i in range(len(orig_graph)):
            orig_graph[i].x = torch.nn.functional.normalize(
                orig_graph[i].x, p=2.0, dim=1
            )
            orig_graph[i].edge_attr = torch.nn.functional.normalize(
                orig_graph[i].edge_attr, p=2.0, dim=1
            )
        return orig_graph

    def _cal_loss(
        self, global_1, global_2, mu, beta, gum_matrix_nodes, perm_matrix, device
    ):
        """
        Calculate Hamming Loss for the graph matching component between graph 1 and graph 2.
        """
        loss = 0
        orig_graph = self.build_graph(global_1, global_2)
        unary_costs_list = [
            self.vertex_affinity(g_1.x, g_2.x) for (g_1, g_2) in lexico_iter(orig_graph)
        ]
        unary_costs_list[0] += gum_matrix_nodes
        unary_costs_list[0] -= 0.5
        unary_costs_list[0] *= -1

        quadratic_costs_list = [
            self.edge_affinity(g_1.edge_attr, g_2.edge_attr)
            for (g_1, g_2) in lexico_iter(orig_graph)
        ]
        num_edges = orig_graph[0].edge_attr.shape[0]
        quadratic_costs_list[0] += torch.from_numpy(
            np.random.gumbel(mu, beta, (num_edges, num_edges))
        ).to(device)
        quadratic_costs_list[0] -= 0.5
        quadratic_costs_list[0] *= -0.5
        gm_solvers = GraphMatchingModule(
            [orig_graph[0].edge_index],
            [orig_graph[1].edge_index],
            [global_1[0].shape[0]],
            [global_1[0].shape[0]],
            cfg.BB_GM.lambda_val,
            cfg.BB_GM.solver_params,
        )
        matching = gm_solvers(unary_costs_list, quadratic_costs_list)
        loss = self.hamming_loss(matching[0], perm_matrix)
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        merge_sentence_id: torch.LongTensor = None,  ## old name: Fsentences_id -> deprecation for linting
        merge_sentence_id_long: torch.LongTensor = None,
        attention_mask_F: Optional[torch.Tensor] = None,
        attention_mask_F_long: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        Fsentences_embeds = None

        # Extract word embedding of both input for autoregressive and multi-graph matching modules
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.contrastive:
                Fsentences_embeds = self.embed_tokens(merge_sentence_id)
                if self.multi_graph:
                    Fsentences_embeds_long = self.embed_tokens(merge_sentence_id_long)

        vision_tower = getattr(self, "vision_tower", None)

        lossAlign = 0
        if (
            vision_tower is not None
            and (input_ids.shape[1] != 1 or self.training)
            and images is not None
        ):
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_feature, dummy_image_features = (
                            self.extract_visual_features(
                                vision_tower, image.unsqueeze(0)
                            )
                        )
                        image_features.append(image_feature)
                else:
                    image_features, dummy_image_features = self.extract_visual_features(
                        vision_tower, images
                    )

            if type(images) is list:
                image_features = [
                    self.mm_projector(image_feature)[0]
                    for image_feature in image_features
                ]
            else:
                image_features = self.mm_projector(image_features)

            dummy_image_features = self.mm_projector(dummy_image_features)

            # If using contrastive, check whether using language embedding before or after the LLM
            if self.contrastive:
                if self.after_de:
                    align_input_embeds = super(LlavaLlamaModelBase, self).forward(
                        input_ids=None,
                        attention_mask=attention_mask_F,
                        past_key_values=past_key_values,
                        inputs_embeds=Fsentences_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )[0]
                    if self.multi_graph:
                        align_input_embeds_long = super(
                            LlavaLlamaModelBase, self
                        ).forward(
                            input_ids=None,
                            attention_mask=attention_mask_F_long,
                            past_key_values=past_key_values,
                            inputs_embeds=Fsentences_embeds_long,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                        )[
                            0
                        ]
                else:
                    align_input_embeds = Fsentences_embeds
                    if self.multi_graph:
                        align_input_embeds_long = Fsentences_embeds_long

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_image_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (
                        cur_input_ids == vision_tower.config.im_end_token
                    ).sum():
                        raise ValueError(
                            "The number of image start tokens and image end tokens should be the same."
                        )
                    image_start_tokens = torch.where(
                        cur_input_ids == vision_tower.config.im_start_token
                    )[0]
                    image_end_tokens = torch.where(
                        cur_input_ids == vision_tower.config.im_end_token
                    )[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(
                            device=cur_input_embeds.device
                        )
                        num_patches = cur_image_features.shape[0]

                        if (
                            cur_input_ids[image_start_token_pos + num_patches + 1]
                            != vision_tower.config.im_end_token
                        ):
                            raise ValueError(
                                "The image end token should follow the image start token."
                            )
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:image_start_token_pos].detach(),
                                    cur_input_embeds[
                                        image_start_token_pos : image_start_token_pos
                                        + 1
                                    ],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos
                                        + num_patches
                                        + 1 : image_start_token_pos
                                        + num_patches
                                        + 2
                                    ],
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 2 :
                                    ].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[: image_start_token_pos + 1],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 1 :
                                    ],
                                ),
                                dim=0,
                            )
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (
                        cur_input_ids == vision_tower.config.im_patch_token
                    ).sum() != num_patches:
                        raise ValueError(
                            "The number of image patch tokens should be the same as the number of image patches."
                        )
                    masked_indices = torch.where(
                        cur_input_ids == vision_tower.config.im_patch_token
                    )[0]
                    mask_index_start = masked_indices[0]
                    if (
                        masked_indices
                        != torch.arange(
                            mask_index_start,
                            mask_index_start + num_patches,
                            device=masked_indices.device,
                            dtype=masked_indices.dtype,
                        )
                    ).any():
                        raise ValueError(
                            "The image patch tokens should be consecutive."
                        )
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_image_features,
                                cur_input_embeds[
                                    mask_index_start + num_patches :
                                ].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_image_features,
                                cur_input_embeds[mask_index_start + num_patches :],
                            ),
                            dim=0,
                        )
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

            """
            Adding Contrastive Mechanism
            """

            if self.contrastive:

                align_image_features = image_features.clone()

                if self.multi_graph:

                    logger.info("Using Multi-Graph")
                    align_image_features_mean = align_image_features.mean(1)
                    align_input_embeds_mean = align_input_embeds.mean(1)
                    align_input_embeds_long_mean = align_input_embeds_long.mean(1)
                    barycenter = (
                        1.0
                        * (
                            align_image_features_mean
                            + align_input_embeds_mean
                            + align_input_embeds_long_mean
                        )
                        / 3
                    )

                    mu, beta = 0, 0.1 / 2
                    gum_matrix_nodes = torch.from_numpy(
                        np.random.gumbel(
                            mu,
                            beta,
                            (
                                align_image_features.shape[0],
                                align_image_features.shape[0],
                            ),
                        )
                    ).to(image_features.device)

                    perm_matrix = torch.diag(
                        torch.ones(align_image_features.shape[0]), 0
                    ).to(image_features.device)

                    lossAlign = 0

                    for each_global in [
                        (align_image_features_mean, "none"),
                        (align_input_embeds_mean, "none"),
                        (align_input_embeds_long_mean, "none"),
                    ]:
                        print(each_global[0].shape)
                        loss = 0
                        loss += self._cal_loss(
                            each_global,
                            (barycenter, "none"),
                            mu,
                            beta,
                            gum_matrix_nodes,
                            perm_matrix,
                            image_features.device,
                        )
                        loss += self._cal_loss(
                            (barycenter, "none"),
                            each_global,
                            mu,
                            beta,
                            gum_matrix_nodes,
                            perm_matrix,
                            image_features.device,
                        )
                        loss = loss / 2
                        lossAlign += loss

                    lossAlign /= 3

        if not hasattr(self, "alpha"):
            self.alpha = 1.0

        # print("inputs_embed shape:", inputs_embeds.shape)
        return (
            super(LlavaLlamaModelBase, self).forward(
                input_ids=None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ),
            lossAlign,
            self.alpha,
            self.grad_step,
        )


class LlavaLlamaForCausalLM(LlavaLlamaForCausalLMBase):
    def __init__(self, config):
        super(LlavaLlamaForCausalLMBase, self).__init__(config)
        self.iter = 0
        self.lossGen = []
        self.lossCon = []
        self.model = LlavaLlamaModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        merge_sentence_id: torch.LongTensor = None,  ## modified
        merge_sentence_id_long: torch.LongTensor = None,
        attention_mask_F: Optional[torch.Tensor] = None,  ## modified
        attention_mask_F_long: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, lossAlign, alphaArg, grad_step = self.model(
            input_ids=input_ids,
            merge_sentence_id=merge_sentence_id,  ## modified
            merge_sentence_id_long=merge_sentence_id_long,
            attention_mask_F=attention_mask_F,  ## modified
            attention_mask_F_long=attention_mask_F_long,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        decoder_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        self.lossGen.append(decoder_output["loss"].cpu().item())
        logger.info(
            f"Loss: \n \t alpha: {alphaArg} | decoder_output_loss: {decoder_output['loss'].cpu().item()} | lossAlign: {lossAlign}"
        )

        if lossAlign != 0:
            self.lossCon.append((lossAlign).cpu().item())
        else:
            self.lossCon.append(lossAlign)

        if len(self.lossGen) == grad_step:
            with open(os.path.join(LOG_DIR, "Contras.txt"), "a") as f:
                f.write(str(self.iter) + " " + str(sum(self.lossCon)) + "\n")
            with open(os.path.join(LOG_DIR, "Gen.txt"), "a") as f:
                f.write(str(self.iter) + " " + str(sum(self.lossGen)) + "\n")

            self.iter += 1
            self.lossGen.clear()
            self.lossCon.clear()

        return (
            decoder_output["loss"] + alphaArg * lossAlign,
            (lossAlign, decoder_output["loss"]),
        )

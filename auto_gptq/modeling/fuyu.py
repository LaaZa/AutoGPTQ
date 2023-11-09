from ._base import *


class FuyuGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "FuyuDecoderLayer"
    layers_block_name = "language_model.model.layers"
    outside_layer_modules = ["language_model.model.embed_tokens", "language_model.model.final_layernorm", "vision_embed_tokens"]
    inside_layer_modules = [
        ["self_attn.query_key_value"],
        ["self_attn.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]


__all__ = ["FuyuGPTQForCausalLM"]
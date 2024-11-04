import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.nn.init as init
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from safetensors.torch import load_file
import json
from dataclasses import dataclass

normalize = (
    lambda u, threshold: (
        (u.abs() >= threshold).float()
        * ((u.abs().log().nan_to_num(0) + threshold) * u.sign())
    )
    + (u.abs() < threshold).float() * u
)
denormalize = (
    lambda u, threshold: (
        (u.abs() >= threshold).float()
        * (((u.abs() - threshold).exp().nan_to_num(0)) * u.sign())
    )
    + (u.abs() < threshold).float() * u
)

@dataclass
class EstiFormerConfig:
    hidden_size: int = 128
    num_outputs: int = 4
    num_heads: int = 8
    max_data_scale: float = 100.0
    num_layers: int = 4
    num_inducer: int = 64
    estimator_multiplier: int = 4

class EstiFormerMultiChannelInputLayer(nn.Module):
    def __init__(
        self,
        num_channels,
        dim_hidden,
    ):
        super(EstiFormerMultiChannelInputLayer, self).__init__()
        self.x_scale_input_layer = nn.Sequential(
            nn.Linear(1, dim_hidden), nn.LayerNorm(dim_hidden), nn.GELU()
        )
        self.x_norm_input_layer = nn.Sequential(
            nn.Linear(1, dim_hidden), nn.LayerNorm(dim_hidden), nn.GELU()
        )

        self.channel_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, dim_hidden), nn.LayerNorm(dim_hidden), nn.GELU()
                )
                for _ in range(num_channels)
            ]
        )
        self.channel_modulator = nn.Sequential(
            nn.Linear(dim_hidden * num_channels, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
        )

    def forward(self, *x):
        x = torch.cat([c(x_) for c, x_ in zip(self.channel_encoders, x)], -1)
        x = self.channel_modulator(x)
        return x


class EstiformerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        bias = True
        dropout = 0.0
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, hidden_states, key_value_states):
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class EstiFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        num_inducers=128,
    ):
        super(EstiFormerEncoderLayer, self).__init__()
        self.inducer = nn.Parameter(torch.Tensor(1, num_inducers, hidden_size))
        nn.init.xavier_uniform_(self.inducer)
        self.input_inducer_attn = EstiformerAttention(hidden_size, num_heads)
        self.inducer_output_attn = EstiformerAttention(hidden_size, num_heads)
        self.ln0 = nn.LayerNorm(hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        b = x.size(0)
        residual = x
        x = self.ln0(x)
        mem = self.input_inducer_attn(self.inducer.repeat(b, 1, 1), x)
        x = self.inducer_output_attn(x, mem) + residual
        residual = x
        x = self.ln1(x)
        x = self.dropout(self.ffn(x)) + residual
        return x


class EstiFormerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        num_inducers=128,
        num_layers=4,
    ):
        super(EstiFormerEncoder, self).__init__()
        self.layers = nn.Sequential(
            *[
                EstiFormerEncoderLayer(hidden_size, num_heads, num_inducers)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        return self.layers(x)


class EstiFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        num_inducers=128,
    ):

        super(EstiFormerDecoderLayer, self).__init__()
        self.estimator_self_attn = EstiformerAttention(hidden_size, num_heads)
        self.self_attn_ln = nn.LayerNorm(hidden_size)

        self.inducer = nn.Parameter(torch.Tensor(1, num_inducers, hidden_size))
        nn.init.xavier_uniform_(self.inducer)
        self.input_inducer_attn = EstiformerAttention(hidden_size, num_heads)
        self.inducer_output_attn = EstiformerAttention(hidden_size, num_heads)
        self.encoder_decoder_ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.final_ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        b = x.size(0)

        residual = x
        x = self.encoder_decoder_ln(x)

        mem = self.input_inducer_attn(self.inducer.repeat(b, 1, 1), y)
        x = self.inducer_output_attn(x, mem)
        x = self.dropout(x) + residual

        residual = x
        x = self.self_attn_ln(x)
        x = self.estimator_self_attn(x, x)
        x = self.dropout(x) + residual

        residual = x
        x = self.final_ln(x)
        x = self.dropout(self.ffn(x)) + residual
        return x


class EstiFormerDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        num_inducers=128,
        num_layers=4,
    ):
        super(EstiFormerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EstiFormerDecoderLayer(hidden_size, num_heads, num_inducers)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x, y)
        return x


class EstiFormer(nn.Module):
    def __init__(
        self,
        config: EstiFormerConfig,
    ):
        super(EstiFormer, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_outputs = config.num_outputs
        self.num_heads = config.num_heads
        self.max_data_scale = config.max_data_scale
        self.num_layers = config.num_layers
        self.num_inducer = config.num_inducer
        self.max_data_scale = config.max_data_scale
        self.estimator_multiplier = config.estimator_multiplier

        self.multichannel_input_layer = EstiFormerMultiChannelInputLayer(
            2, self.hidden_size
        )
        self.encoder = EstiFormerEncoder(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_inducers=self.num_inducer,
            num_layers=self.num_layers,
        )
        self.decoder = EstiFormerDecoder(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_inducers=self.num_inducer,
            num_layers=self.num_layers,
        )

        self.num_estimator = self.num_outputs
        self.estimator_embedding = nn.Embedding(
            self.num_estimator * self.estimator_multiplier, self.hidden_size
        )

        self.predictors = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(self.num_outputs)]
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)

    @torch.no_grad()
    def prepare_input(self, x):
        x_mean = x.mean(-1, keepdims=True)
        x_std = x.std(-1, keepdims=True)
        x_mean = normalize(x_mean, self.max_data_scale)
        x_std = normalize(x_std, self.max_data_scale)
        x_norm = (x - x_mean) / x_std
        x_scaled = normalize(x, self.max_data_scale)
        return x_norm, x_scaled, x_mean, x_std

    def forward(self, x):
        b, n = x.shape
        with torch.no_grad():
            x_norm, x_scaled, x_mean, x_std = self.prepare_input(x)
        x_norm = x_norm.view(b, n, 1)
        x_scaled = x_scaled.view(b, n, 1)
        x = self.multichannel_input_layer(x_norm, x_scaled)
        estimator = self.estimator_embedding.weight.data.repeat(b, 1, 1)

        x = self.encoder(x)
        decoder_out = self.decoder(estimator, x)

        decoder_out = decoder_out.view(
            x.size(0), self.num_estimator, self.estimator_multiplier, -1
        ).mean(2)
        out = torch.cat(
            [p(decoder_out[:, i, :]) for i, p in enumerate(self.predictors)], -1
        ).view(x.size(0), -1)
        estimation = torch.concat(
            [
                out[:, 0:1] * x_std + x_mean,
                out[:, 1:2] * x_std,
                out[:, 2:],
            ],
            -1,
        ).contiguous()
        return estimation

    @classmethod
    def from_pretrained(self, model_path):
        config_json = json.load(open(os.path.join(model_path, "config.json")))
        config = EstiFormerConfig(**config_json)
        model = EstiFormer(config)
        ckpt = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(ckpt)
        return model

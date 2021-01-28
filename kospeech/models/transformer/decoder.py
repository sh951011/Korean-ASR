# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from kospeech.models.transformer.sublayers import AddNorm
from kospeech.models.attention import MultiHeadAttention
from kospeech.models.decoder import IncrementalDecoder
from kospeech.models.transformer.embeddings import Embedding, PositionalEncoding
from kospeech.models.transformer.mask import get_decoder_self_attn_mask, get_attn_pad_mask
from kospeech.models.transformer.sublayers import PositionwiseFeedForwardNet


class SpeechTransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            ffnet_style: str = 'ff',        # style of feed forward network
    ) -> None:
        super(SpeechTransformerDecoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.memory_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionwiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(
            self,
            inputs: Tensor,                                 # B x T_input
            memory: Tensor,                                 # B x T_input x D_model
            self_attn_mask: Optional[Tensor] = None,        # B x T_input x T_input
            memory_mask: Optional[Tensor] = None,           # B x T_input x T_output
    ) -> Tuple[Tensor, Tensor, Tensor]:
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs, memory_attn = self.memory_attention(outputs, memory, memory, memory_mask)
        outputs = self.feed_forward(outputs)
        return outputs, self_attn, memory_attn


class SpeechTransformerDecoder(IncrementalDecoder):
    """
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.

    Args:
        num_classes: umber of classes
        d_model: dimension of model
        d_ff: dimension of feed forward network
        num_layers: number of decoder layers
        num_heads: number of attention heads
        ffnet_style: style of feed forward network
        dropout_p: probability of dropout
        pad_id: identification of pad token
        eos_id: identification of end of sentence token
    """

    def __init__(
            self,
            num_classes: int,               # number of classes
            d_model: int = 512,             # dimension of model
            d_ff: int = 512,                # dimension of feed forward network
            num_layers: int = 6,            # number of decoder layers
            num_heads: int = 8,             # number of attention heads
            ffnet_style: str = 'ff',        # style of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            pad_id: int = 0,                # identification of pad token
            sos_id: int = 1,                # identification of start of sentence token
            eos_id: int = 2,                # identification of end of sentence token
            max_length: int = 400,          # max length for decoding
            device: torch.device = 'cuda',  # device 'cuda' or 'cpu'
    ) -> None:
        super(SpeechTransformerDecoder, self).__init__(
            num_classes=num_classes, max_length=max_length, device=device,
            pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
        )
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            SpeechTransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
                ffnet_style=ffnet_style,
            ) for _ in range(num_layers)
        ])

    def forward_step(
            self,
            prev_step_outputs: Tensor,
            encoder_output_lengths: Tensor,
            encoder_outputs: Tensor
    ) -> Tensor:
        step_output = self.decoder(prev_step_outputs, encoder_output_lengths, encoder_outputs)
        step_output = self.fc(step_output)
        return step_output.max(dim=-1, keepdim=False)[1]

    def forward(
            self,
            inputs: Tensor,
            encoder_output_lengths: Tensor,
            encoder_outputs: Tensor = None
    ) -> Tensor:
        batch_size, target_length = inputs.size(0), inputs.size(1)

        self_attn_mask = get_decoder_self_attn_mask(inputs, inputs, self.pad_id)
        encoder_outputs_mask = get_attn_pad_mask(encoder_outputs, encoder_output_lengths, target_length)

        outputs = self.embedding(inputs) + self.positional_encoding(target_length)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(outputs, encoder_outputs, self_attn_mask, encoder_outputs_mask)

        return outputs

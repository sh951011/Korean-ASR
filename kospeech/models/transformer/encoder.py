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

import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor

from kospeech.models.transformer.sublayers import AddNorm
from kospeech.models.attention import MultiHeadAttention
from kospeech.models.encoder import ConvolutionalEncoder
from kospeech.models.modules import Linear, LayerNorm, Transpose
from kospeech.models.transformer.embeddings import PositionalEncoding
from kospeech.models.transformer.mask import get_attn_pad_mask
from kospeech.models.transformer.sublayers import PositionwiseFeedForwardNet


class SpeechTransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".

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
            ffnet_style: str = 'ff'         # style of feed forward network
    ) -> None:
        super(SpeechTransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionwiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs: Tensor, self_attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs = self.feed_forward(outputs)
        return outputs, attn


class SpeechTransformerEncoder(ConvolutionalEncoder):
    """
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        d_model: dimension of model (default: 512)
        input_dim: dimension of feature vector (default: 80)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoder layers (default: 6)
        num_heads: number of attention heads (default: 8)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
        dropout_p:  probability of dropout (default: 0.3)
        pad_id: identification of pad token (default: 0)

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths
    """

    def __init__(
            self,
            input_dim: int = 80,                 # dimension of feature vector
            num_classes: int = None,             # number of classfication
            d_model: int = 512,                  # dimension of model
            d_ff: int = 2048,                    # dimension of feed forward network
            num_layers: int = 6,                 # number of encoder layers
            num_heads: int = 8,                  # number of attention heads
            ffnet_style: str = 'ff',             # style of feed forward network [ff, conv]
            extractor: str = 'vgg',              # type of CNN extractor
            activation: str = 'relu',            # type of activation function
            mask_conv: bool = False,             # flag indication whether apply mask convolution or not
            dropout_p: float = 0.3,              # probability of dropout
            pad_id: int = 0,                     # identification of pad token
            joint_ctc_attention: bool = False    # flag indication whether to apply joint ctc attention
    ) -> None:
        self.extractor = extractor
        super(SpeechTransformerEncoder, self).__init__(
            input_dim, extractor=self.extractor, activation=activation, mask_conv=mask_conv,
        )
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_id = pad_id
        self.input_proj = Linear(self.get_conv_output_dim(), d_model)
        self.input_layer_norm = LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            SpeechTransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
                ffnet_style=ffnet_style,
            ) for _ in range(num_layers)
        ])
        self.joint_ctc_attention = joint_ctc_attention
        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                nn.BatchNorm1d(d_model),
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(d_model, num_classes, bias=False),
            )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        encoder_log_probs = None

        inputs, encoder_output_lengths = self.conv_forward(inputs, input_lengths)
        max_seq_length = inputs.size(1)
        self_attn_mask = get_attn_pad_mask(inputs, encoder_output_lengths, max_seq_length)

        encoder_outputs = self.input_layer_norm(self.input_proj(inputs)) + self.positional_encoding(max_seq_length)
        encoder_outputs = self.input_dropout(encoder_outputs)

        for layer in self.layers:
            encoder_outputs, attn = layer(encoder_outputs, self_attn_mask)

        if self.joint_ctc_attention:
            encoder_outputs = self.fc(encoder_outputs.transpose(1, 2))
            encoder_log_probs = F.log_softmax(encoder_outputs, dim=-1)

        return encoder_outputs, encoder_log_probs, encoder_output_lengths

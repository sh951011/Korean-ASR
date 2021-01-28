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
from typing import Tuple

from kospeech.models.encoder import ConvolutionalCTCModel
from kospeech.models.modules import Linear, BNReluRNN


class DeepSpeech2(ConvolutionalCTCModel):
    """
    Deep Speech2 model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1512.02595

    Args:
        input_dim (int): dimension of input
        num_classes (int): number of classfication
        rnn_type (str, optional): type of RNN cell (default: gru)
        num_rnn_layers (int, optional): number of recurrent layers (default: 5)
        rnn_hidden_dim (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability (default: 0.1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        activation (str): type of activation function (default: hardtanh)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence
    """
    def __init__(
            self,
            input_dim: int,                         # size of input
            num_classes: int,                       # number of classfication
            rnn_type='gru',                         # type of RNN cell
            extractor: str = 'ds2',                 # type of CNN extractor
            num_rnn_layers: int = 5,                # number of RNN layers
            rnn_hidden_dim: int = 512,              # dimension of RNN`s hidden state
            dropout_p: float = 0.1,                 # dropout probability
            bidirectional: bool = True,             # if True, becomes a bidirectional rnn
            activation: str = 'hardtanh',           # type of activation function
            device: torch.device = 'cuda',          # device - 'cuda' or 'cpu'
    ):
        super(DeepSpeech2, self).__init__(
            input_dim=input_dim, in_channels=1, out_channels=32,
            extractor=extractor, activation=activation, mask_conv=True,
        )
        self.rnn_layers = nn.ModuleList()
        self.device = device

        conv_output_dim = self.get_conv_output_dim()
        rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim

        for idx in range(num_rnn_layers):
            self.rnn_layers.append(BNReluRNN(
                input_size=conv_output_dim if idx == 0 else rnn_output_size,
                hidden_dim=rnn_hidden_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                dropout_p=dropout_p,
                device=device,
            ).to(self.device))

        self.fc = nn.Sequential(
            Linear(rnn_output_size, rnn_hidden_dim),
            nn.ReLU(),
            Linear(rnn_hidden_dim, num_classes, bias=False),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        outputs, output_lengths = self.conv_forward(inputs, input_lengths)

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = self.get_normalized_probs(outputs.transpose(0, 1))

        return outputs, output_lengths

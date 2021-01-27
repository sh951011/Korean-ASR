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
from torch import Tensor
from typing import Tuple, Optional

from kospeech.models import supported_rnns
from kospeech.models.encoder import ConvolutionalEncoder
from kospeech.models.modules import Linear, Transpose


class Listener(ConvolutionalEncoder):
    """
    Converts low level speech signals into higher level features

    Args:
        input_dim (int): dimension of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability (default: 0.3)
        extractor (str): type of CNN extractor (default: vgg)
        activation (str): type of activation function (default: hardtanh)
        mask_conv (bool): flag indication whether apply mask convolution or not
        joint_ctc_attention (bool): flag indication whether to apply joint ctc attention

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: encoder_outputs, hidden
        - **encoder_outputs**: tensor containing the encoded features of the input sequence
        - **encoder_log__probs**: tensor containing log probability for ctc loss
    """

    def __init__(
            self,
            input_dim: int,                          # size of input
            num_classes: int = None,                 # number of class
            hidden_dim: int = 512,                   # dimension of RNN`s hidden state
            dropout_p: float = 0.3,                  # dropout probability
            num_layers: int = 3,                     # number of RNN layers
            bidirectional: bool = True,              # if True, becomes a bidirectional encoder
            rnn_type: str = 'lstm',                  # type of RNN cell
            extractor: str = 'vgg',                  # type of CNN extractor
            activation: str = 'hardtanh',            # type of activation function
            mask_conv: bool = False,                 # flag indication whether apply mask convolution or not
            joint_ctc_attention: bool = False,       # flag indication whether to apply joint ctc attention
    ) -> None:
        self.extractor = extractor.lower()
        self.mask_conv = mask_conv
        super(Listener, self).__init__(input_dim, extractor=self.extractor, activation=activation, mask_conv=mask_conv)
        rnn_cell = supported_rnns[rnn_type]
        self.rnn = rnn_cell(self.get_conv_output_dim(), hidden_dim, num_layers, True, True, dropout_p, bidirectional)
        self.hidden_dim = hidden_dim
        self.joint_ctc_attention = joint_ctc_attention

        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                nn.BatchNorm1d(self.hidden_dim << 1),
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(hidden_dim << 1, num_classes, bias=False),
            )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        encoder_log_probs = None

        conv_outputs, encoder_output_lengths = self.conv_forward(inputs, input_lengths)
        conv_outputs = conv_outputs.permute(1, 0, 2).contiguous()

        conv_outputs = nn.utils.rnn.pack_padded_sequence(conv_outputs, encoder_output_lengths.cpu())
        encoder_outputs, hidden = self.rnn(conv_outputs)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        if self.joint_ctc_attention:
            encoder_outputs = self.fc(encoder_outputs.transpose(1, 2))
            encoder_log_probs = F.log_softmax(encoder_outputs, dim=-1)

        return encoder_outputs, encoder_log_probs, encoder_output_lengths

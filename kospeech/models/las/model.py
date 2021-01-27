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

from kospeech.models import TopKDecoder
from kospeech.models.model import EncoderDecoderModel


class ListenAttendSpell(EncoderDecoderModel):
    """
    Listen, Attend and Spell model with configurable encoder and decoder.

    Args:
        encoder (torch.nn.Module): encoder of las
        decoder (torch.nn.Module): decoder of las

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)

    Returns: output
        - **output** (seq_len, batch_size, num_classes): list of tensors containing
          the outputs of the decoding function.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(ListenAttendSpell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            inputs: Tensor,                             # tensor contains audio's features
            input_lengths: Tensor,                      # tensor contains feature's lengths
            targets: Optional[Tensor] = None,           # tensor contains target sentences
            teacher_forcing_ratio: float = 1.0,         # ratio of teacher forcing
    ) -> Tuple[dict, Tensor, Tensor]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        encoder_outputs, encoder_log_probs, encoder_output_lengths = self.encoder(inputs, input_lengths)

        if isinstance(self.decoder, TopKDecoder):
            return self.decoder(targets, encoder_outputs)
        decoder_outputs = self.decoder(targets, encoder_outputs, teacher_forcing_ratio)

        return decoder_outputs, encoder_log_probs, encoder_output_lengths

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor, device: str) -> Tensor:
        with torch.no_grad():
            self.flatten_parameters()
            decoder_outputs, _, _ = self.forward(inputs, input_lengths, teacher_forcing_ratio=0.0)
            outputs = torch.stack(decoder_outputs['decoder_log_probs'], dim=1).to(device)
            return outputs.max(-1)[1]

    def flatten_parameters(self) -> None:
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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

from kospeech.models.jasper import JasperEncoderConfig
from kospeech.models.conv import MaskConv1d
from kospeech.models.jasper.sublayers import (
    JasperSubBlock, 
    JasperBlock,
)


class JasperEncoder(nn.Module):
    """
    Jasper Encoder consists of one pre-processing blocks and B Jasper blocks.

    Args:
        config (JasperEncoderConfig): configurations of Jasper Encoder
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """

    def __init__(self, config: JasperEncoderConfig, device: torch.device) -> None:
        super(JasperEncoder, self).__init__()
        self.config = config
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(
            JasperSubBlock(
                in_channels=self.config.preprocess_block['in_channels'],
                out_channels=self.config.preprocess_block['out_channels'],
                kernel_size=self.config.preprocess_block['kernel_size'],
                stride=self.config.preprocess_block['stride'],
                dilation=self.config.preprocess_block['dilation'],
                dropout_p=self.config.preprocess_block['dropout_p'],
                activation='relu',
                bias=False,
            ).to(self.device)
        )
        self.layers.extend([
            JasperBlock(
                num_sub_blocks=self.config.num_sub_blocks,
                in_channels=self.config.block['in_channels'][i],
                out_channels=self.config.block['out_channels'][i],
                kernel_size=self.config.block['kernel_size'][i],
                dilation=self.config.block['dilation'][i],
                dropout_p=self.config.block['dropout_p'][i],
                activation='relu',
                bias=False,
            ).to(self.device) for i in range(config.num_blocks)
        ])
        self.residual_connections = self._create_jasper_dense_residual_connections(self.config.num_blocks)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        inputs (torch.FloatTensor): (batch_size, dimension, sequence_length)
        input_lengths (torch.LongTensor): (batch_size)
        """
        prev_outputs, prev_output_lengths = list(), list()
        residual = None

        for i, layer in enumerate(self.layers[:-1]):
            inputs, input_lengths = layer(inputs, input_lengths, residual)
            prev_outputs.append(inputs)
            prev_output_lengths.append(input_lengths)
            residual = self._get_jasper_dencse_residual(prev_outputs, prev_output_lengths, i)

        outputs, output_lengths = self.layers[-1](inputs, input_lengths, residual)
        del prev_outputs, prev_output_lengths, residual, inputs, input_lengths

        return outputs, output_lengths

    def _get_jasper_dencse_residual(self, prev_outputs: list, prev_output_lengths: list, index: int):
        residual = None

        for item in zip(prev_outputs, prev_output_lengths, self.residual_connections[index]):
            prev_output, prev_output_length, residual_modules = item
            conv1x1, batch_norm = residual_modules

            if residual is None:
                residual = conv1x1(prev_output, prev_output_length)[0]
            else:
                residual += conv1x1(prev_output, prev_output_length)[0]

            residual = batch_norm(residual)

        return residual

    def _create_jasper_dense_residual_connections(self, num_blocks: int) -> nn.ModuleList:
        residual_connections = nn.ModuleList()

        for i in range(num_blocks):
            residual_modules = nn.ModuleList()
            for j in range(i + 1):
                residual_modules.append(nn.ModuleList([
                    MaskConv1d(self.config.block['in_channels'][j], self.config.block['out_channels'][i], kernel_size=1),
                    nn.BatchNorm1d(self.config.block['out_channels'][i], eps=1e-03, momentum=0.1)
                ]))
            residual_connections.append(residual_modules)

        return residual_connections

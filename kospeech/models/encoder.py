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

import math
import torch
from torch import Tensor
from typing import Tuple

from kospeech.models.conv import VGGExtractor, DeepSpeech2Extractor, Conv2dSubsampling
from kospeech.models.model import BaseModel


class BaseEncoder(BaseModel):
    """
    Base encoder class. Specifies the interface used by different encoder types.
    """

    def __init__(self):
        super(BaseEncoder, self).__init__()

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


class ConvolutionalEncoder(BaseEncoder):

    def __init__(
            self,
            input_dim: int,
            in_channels: int = 1,
            out_channels: int or tuple = None,
            extractor: str = 'conv2d',
            activation: str = 'relu',
            mask_conv: bool = False,
    ) -> None:
        super(ConvolutionalEncoder, self).__init__()
        self.extractor = extractor.lower()
        self.input_dim = input_dim

        if out_channels is None:
            if self.extractor == 'vgg':
                out_channels = (64, 128)
            elif self.extractor in ('ds2', 'conv2d'):
                out_channels = 32

        if self.extractor == 'vgg':
            assert isinstance(out_channels, tuple), "VGGExtractor's out_channels should be instance of tuple"
            assert len(out_channels) == 2, "VGGExtractor's out_channels should be a tuple of 2"

            self.conv = VGGExtractor(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation,
                mask_conv=mask_conv,
            )
        elif self.extractor == 'ds2':
            assert isinstance(out_channels, int), "DeepSpeech2Extractor's out_channels should be instance of int"

            self.conv = DeepSpeech2Extractor(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation,
                mask_conv=mask_conv,
            )
        elif self.extractor == 'conv2d':
            self.conv = Conv2dSubsampling(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation,
            )
        else:
            raise ValueError(f"Unsupported Extractor : {self.extractor}")

    def conv_forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        return self.conv(inputs, input_lengths)

    def get_conv_output_dim(self):
        if self.extractor == 'vgg':
            output_dim = (self.input_dim - 1) << 5 if self.input_dim % 2 else self.input_dim << 5

        elif self.extractor == 'ds2':
            output_dim = int(math.floor(self.input_dim + 2 * 20 - 41) / 2 + 1)
            output_dim = int(math.floor(output_dim + 2 * 10 - 21) / 2 + 1)
            output_dim <<= 5

        elif self.extractor == 'conv2d':
            output_dim = ((self.input_dim - 1) // 2 - 1) // 2

        else:
            raise ValueError(f"Unsupported Extractor : {self.extractor}")

        return output_dim

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


class ConvolutionalCTCModel(ConvolutionalEncoder):
    def __init__(
            self,
            input_dim: int,
            in_channels: int = 1,
            out_channels: int or tuple = None,
            extractor: str = 'conv2d',
            activation: str = 'relu',
            mask_conv: bool = False,
    ) -> None:
        super(ConvolutionalCTCModel, self).__init__(
            input_dim=input_dim, in_channels=in_channels, out_channels=out_channels,
            extractor=extractor, activation=activation, mask_conv=mask_conv,
        )

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        with torch.no_grad():
            outputs, output_lengths = self.forward(inputs, input_lengths)
            return outputs.max(-1)[1]

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

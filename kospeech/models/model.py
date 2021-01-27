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


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def count_parameters(self):
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout):
        raise NotImplementedError


class EncoderDecoderModel(BaseModel):
    def __init__(self) -> None:
        super(EncoderDecoderModel, self).__init__()
        self.encoder = None
        self.decoder = None

    def set_encoder(self, encoder: nn.Module) -> None:
        self.encoder = encoder

    def set_decoder(self, decoder: nn.Module) -> None:
        self.decoder = decoder

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor, device: torch.device) -> Tensor:
        raise NotImplementedError


class CTCModel(BaseModel):
    def __init__(self):
        super(CTCModel, self).__init__()

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor, device: torch.device) -> Tensor:
        with torch.no_grad():
            outputs, output_lengths = self.forward(inputs, input_lengths)
            return outputs.max(-1)[1]

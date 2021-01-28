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
import torch.nn.functional as F


class BaseDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            max_length: int,
            device: torch.device,
            sos_id: int,
            eos_id: int,
            pad_id: int,
    ):
        super(BaseDecoder, self).__init__()
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id


class IncrementalDecoder(BaseDecoder):
    def __init__(
            self,
            num_classes: int,
            max_length: int,
            device: torch.device,
            sos_id: int,
            eos_id: int,
            pad_id: int,
    ):
        super(IncrementalDecoder, self).__init__(
            num_classes=num_classes, max_length=max_length, device=device,
            sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
        )
        self.beam_size = 1

    def get_normalized_probs(self, outputs):
        outputs = self.fc(outputs)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def set_beam_size(self, beam_size: int):
        assert isinstance(beam_size, int)
        assert beam_size > 0
        self.beam_size = beam_size

    def forward_step(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

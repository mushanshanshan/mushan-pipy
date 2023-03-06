# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa

"""EnCodec neural audio codec."""

__version__ = "0.1.2a3"

from .model import EncodecModel as encodec
from .compress import compress, decompress


def get_24k(bandwidth=24, device='cpu'):
    model = encodec.encodec_model_24khz().to(device)
    model.device = device
    model.set_target_bandwidth(24)
    return model

def get_48k(bandwidth=48, device='cpu'):
    model = encodec.encodec_model_48khz().to(device)
    model.device = device
    model.set_target_bandwidth(24)
    return model
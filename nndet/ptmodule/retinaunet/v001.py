"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from nndet.ptmodule.retinaunet.base import RetinaUNetModule

from nndet.core.boxes.matcher import ATSSMatcher
from nndet.arch.heads.classifier import BCECLassifier
from nndet.arch.heads.regressor import GIoURegressor
from nndet.arch.heads.comb import DetectionHeadHNMNative
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.arch.conv import ConvInstanceRelu, ConvGroupRelu

from nndet.ptmodule import MODULE_REGISTRY

import torch

def load_pretrained_weights(network, fname, verbose=True):
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    for k, value in pretrained_dict.items():
        key = k
        if key.startswith('model.'):
            key = key[6:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)


@MODULE_REGISTRY.register
class RetinaUNetV001(RetinaUNetModule):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = DetectionHeadHNMNative
    head_classifier_cls = BCECLassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg

    @classmethod
    def from_config_plan(cls,
                         model_cfg: dict,
                         plan_arch: dict,
                         plan_anchors: dict,
                         log_num_anchors: str = None,
                         **kwargs,
                         ):
        retinaNet = super().from_config_plan(model_cfg,
                                             plan_arch,
                                             plan_anchors,
                                             log_num_anchors,
                                             **kwargs
                                             )
        if "pretrained_weights" in model_cfg.keys():
            if model_cfg["pretrained_weights"] != "":
                load_pretrained_weights(retinaNet, model_cfg["pretrained_weights"], verbose=True)

        return retinaNet
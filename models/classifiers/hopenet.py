import os
from collections import OrderedDict

import torch
from mmcls.models import (CLASSIFIERS, BaseClassifier, build_backbone,
                          build_head, build_neck)


@CLASSIFIERS.register_module()
class HopeNet(BaseClassifier):
    def __init__(self, backbone, neck, head, pretrained=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained and os.path.exists(pretrained):
            origin_state_dict = torch.load(pretrained)["state_dict"]
            state_dict = OrderedDict()
            for k, v in origin_state_dict.items():
                new_k = k.replace("backbone.", "")
                state_dict[new_k] = v
            self.backbone.load_state_dict(state_dict, strict=False)
            print(F"backbone init from: {pretrained}")
        else:
            print("backbone init from scratch")
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        feature = self.backbone(img)
        if self.with_neck:
            feature = self.neck(feature)
        return feature

    def forward_train(self, img, gt_labels, gt_angles):
        feature = self.extract_feat(img)
        loss = self.head.forward_train(feature, gt_labels, gt_angles)
        return loss

    def simple_test(self, img):
        feature = self.extract_feat(img)
        return self.head.simple_test(feature)

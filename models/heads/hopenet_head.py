import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmcls.models import HEADS
from mmcls.models.heads.base_head import BaseHead


@HEADS.register_module()
class HopenetHead(BaseHead):
    def __init__(self, in_channels, num_bins, alpha, device="cuda"):
        super().__init__()
        self.fc_yaw = nn.Linear(in_channels, num_bins)
        self.fc_pitch = nn.Linear(in_channels, num_bins)
        self.fc_roll = nn.Linear(in_channels, num_bins)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()

        self.idx_tensor = torch.FloatTensor([idx for idx in range(num_bins)])
        if device == "cuda":
            self.idx_tensor = self.idx_tensor.cuda()
        self.alpha = alpha

    def init_weights(self):
        normal_init(self.fc_yaw, mean=0, std=0.01, bias=0)
        normal_init(self.fc_pitch, mean=0, std=0.01, bias=0)
        normal_init(self.fc_roll, mean=0, std=0.01, bias=0)

    def forward_train(self, feature, gt_labels, gt_angles):
        cls_yaw_p = self.fc_yaw(feature)
        cls_pitch_p = self.fc_pitch(feature)
        cls_roll_p = self.fc_roll(feature)

        cls_yaw_g, cls_pitch_g, cls_roll_g = \
            gt_labels[:, 0], gt_labels[:, 1], gt_labels[:, 2]
        cls_yaw_loss = self.cls_criterion(cls_yaw_p, cls_yaw_g)
        cls_pitch_loss = self.cls_criterion(cls_pitch_p, cls_pitch_g)
        cls_roll_loss = self.cls_criterion(cls_roll_p, cls_roll_g)

        reg_yaw_g, reg_pitch_g, reg_roll_g = \
            gt_angles[:, 0], gt_angles[:, 1], gt_angles[:, 2]
        reg_yaw_p = torch.sum(cls_yaw_p * self.idx_tensor, 1) * 3 - 99
        reg_pitch_p = torch.sum(cls_pitch_p * self.idx_tensor, 1) * 3 - 99
        reg_roll_p = torch.sum(cls_roll_p * self.idx_tensor, 1) * 3 - 99
        reg_yaw_loss = self.reg_criterion(reg_yaw_p, reg_yaw_g)
        reg_pitch_loss = self.reg_criterion(reg_pitch_p, reg_pitch_g)
        reg_roll_loss = self.reg_criterion(reg_roll_p, reg_roll_g)

        cls_loss = cls_yaw_loss + cls_pitch_loss + cls_roll_loss
        reg_loss = reg_yaw_loss + reg_pitch_loss + reg_roll_loss
        reg_loss *= self.alpha

        return {"cls_loss": cls_loss, "reg_loss": reg_loss}

    def simple_test(self, feature):
        cls_yaw_p = self.fc_yaw(feature)
        cls_pitch_p = self.fc_pitch(feature)
        cls_roll_p = self.fc_roll(feature)

        reg_yaw_p = torch.sum(cls_yaw_p * self.idx_tensor, 1) * 3 - 99
        reg_pitch_p = torch.sum(cls_pitch_p * self.idx_tensor, 1) * 3 - 99
        reg_roll_p = torch.sum(cls_roll_p * self.idx_tensor, 1) * 3 - 99

        return reg_yaw_p, reg_pitch_p, reg_roll_p

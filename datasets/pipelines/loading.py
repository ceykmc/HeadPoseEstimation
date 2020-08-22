import os

import mmcv
import cv2
import numpy as np
import scipy.io

from mmcls.datasets import PIPELINES
from mmcls.datasets.pipelines import LoadImageFromFile


def read_annotation(anno_file_path: str):
    data = scipy.io.loadmat(anno_file_path)
    # face roi
    landmarks = data["pt2d"]  # 68 points, shape is: [2, 68]
    x_min = min(landmarks[0, :])
    y_min = min(landmarks[1, :])
    x_max = max(landmarks[0, :])
    y_max = max(landmarks[1, :])

    # k = 0.2 to 0.4
    k = np.random.random_sample() * 0.2 + 0.2
    x_min = int(x_min - 0.6 * k * abs(x_max - x_min))
    y_min = int(x_min - 2 * k * abs(y_max - y_min))
    x_max = int(x_max + 0.6 * k * abs(x_max - x_min))
    y_max = int(y_max + 0.6 * k * abs(y_max - y_min))
    roi = [x_min, y_min, x_max, y_max]

    # face pose
    pitch, yaw, roll = data["Pose_Para"][0][:3]
    pitch = pitch * 180 / np.pi
    yaw = yaw * 180 / np.pi
    roll = roll * 180 / np.pi
    angles = [yaw, pitch, roll]  # change order, use (yaw pitch roll)

    return roi, angles


@PIPELINES.register_module()
class LoadImageFromFileWLP(LoadImageFromFile):
    def __init__(self,
                 to_float32=False,
                 color_type="color",
                 file_client_args=dict(backent="disk")):
        super().__init__(to_float32, color_type, file_client_args)

    def __call__(self, results):
        # if self.file_client is None:
            # self.file_client = mmcv.FileClient(**self.file_client_args)

        if results["img_prefix"] is not None:
            filename = os.path.join(results["img_prefix"],
                                    results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        anno_file_path = filename.replace(".jpg", ".mat")
        roi, angles = read_annotation(anno_file_path)

        # img_bytes = self.file_client.get(filename)
        # img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        img = cv2.imread(filename)
        img_h, img_w, img_c = img.shape
        img = img[max(roi[1], 0):min(roi[3], img_h),
                  max(roi[0], 0):min(roi[2], img_w), :]  # face roi

        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)

        bins = np.array(range(-99, 102, 3))
        results["gt_labels"] = np.digitize(angles, bins) - 1
        results["gt_angles"] = np.array(angles, dtype=np.float32)
        return results

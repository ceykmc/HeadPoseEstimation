import argparse
import os
import random
import unittest

import cv2
import numpy as np
from mmcv import Config

import datasets  # noqa: F401,F403
from mmcls.datasets import build_dataset


def argument_parser():
    parser = argparse.ArgumentParser(description="anchors test")
    parser.add_argument("--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--test-method", default="", help="test method name")
    args = parser.parse_args()
    return args


class TestDataset(unittest.TestCase):
    def __init__(self, methodName="runTest", param=None):
        super().__init__(methodName)
        assert "config_file" in param
        self.cfg = Config.fromfile(param["config_file"])

    def show_train_image(self, data, win_name="show"):
        image = data["img"].data.numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image * self.cfg.img_norm_cfg.std + self.cfg.img_norm_cfg.mean
        image = image.astype(dtype=np.uint8, copy=True)
        image = image[:, :, ::-1]  # convert from RGB to BGR
        image = np.ascontiguousarray(image)

        print(F"labels: {data['gt_labels']}, angles: {data['gt_angles']}")
        os.makedirs("./temp", exist_ok=True)
        cv2.imwrite("./temp/test.jpg", image)
        cv2.imshow(win_name, image)
        cv2.waitKey()

    def test_dataset_sample_number(self):
        train_dataset = build_dataset(self.cfg.data.train)
        print(F"train sample number: {len(train_dataset)}")

    def test_show_train_sample(self):
        train_dataset = build_dataset(self.cfg.data.train)
        random_index = random.choice(range(len(train_dataset)))
        print(F"train random index: {random_index}")
        random_train_data = train_dataset[random_index]
        print(F"image shape: {random_train_data['img'].data.shape}")
        self.show_train_image(random_train_data, "train_sample")


def main():
    args = argument_parser()
    config_file = args.config_file
    test_method_name = args.test_method

    if len(test_method_name) == 0:
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(TestDataset)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(TestDataset(name, param={"config_file":
                                                   config_file}))
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(
            TestDataset(test_method_name, param={"config_file": config_file}))
        unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

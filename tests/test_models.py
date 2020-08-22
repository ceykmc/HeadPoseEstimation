import argparse
import unittest

import torch

import models  # noqa: F401,F403
from mmcv import Config
from mmcls.models import build_backbone, build_head, build_neck, build_classifier


class TestModels(unittest.TestCase):
    def __init__(self, methodName="runTest", param=None):
        super().__init__(methodName)
        assert "config_file" in param
        self.cfg = Config.fromfile(param["config_file"])

    def test_backbone(self):
        backbone = build_backbone(self.cfg.model.backbone)
        img_w, img_h = 224, 224
        input_x = torch.randn(4, 3, img_h, img_w)
        outputs = backbone(input_x)
        for i, output in enumerate(outputs):
            print(F"backbone output {i} shape: {output.shape}")

    def test_head(self):
        backbone = build_backbone(self.cfg.model.backbone)
        neck = build_neck(self.cfg.model.neck)
        head = build_head(self.cfg.model.head)
        img_w, img_h = 224, 224
        input_x = torch.randn(4, 3, img_h, img_w)
        feature = backbone(input_x)
        feature = neck(feature)
        yaw, pitch, roll = head.simple_test(feature)
        print(F"yaw: {yaw}, pitch: {pitch}, roll: {roll}")

    def test_classifier(self):
        classifier = build_classifier(self.cfg.model)
        img_w, img_h = 224, 224
        input_x = torch.randn(4, 3, img_h, img_w)
        yaw, pitch, roll = classifier.simple_test(input_x)
        print(F"yaw: {yaw}, pitch: {pitch}, roll: {roll}")


def argument_parser():
    parser = argparse.ArgumentParser(description="anchors test")
    parser.add_argument("--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--test-method", default="", help="test method name")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    config_file = args.config_file
    test_method_name = args.test_method

    if len(test_method_name) == 0:
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(TestModels)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(TestModels(name, param={"config_file": config_file}))
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(
            TestModels(test_method_name, param={"config_file": config_file}))
        unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

import argparse
import unittest

import datasets  # noqa: F401,F403
import models  # noqa: F401,F403
from mmcv import Config
from mmcv.parallel.scatter_gather import scatter
from mmcv.parallel import MMDataParallel
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier


class TestLoss(unittest.TestCase):
    def __init__(self, methodName="runTest", param=None):
        super().__init__(methodName)
        assert "config_file" in param
        self.cfg = Config.fromfile(param["config_file"])

    def read_random_data(self):
        train_dataset = build_dataset(self.cfg.data.train)
        print(F"train sample number: {len(train_dataset)}")
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        data = next(iter(train_dataloader))
        return data

    def test_classifier_loss(self):
        data = self.read_random_data()
        data = scatter(data, ["cuda:0"])
        data = data[0]
        classifier = build_classifier(self.cfg.model)
        classifier = MMDataParallel(module=classifier.cuda(0), device_ids=[0])
        loss = classifier(**data)
        for k, v in loss.items():
            print(k)
            print(v, v.dtype)
            print("---")


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
        testnames = testloader.getTestCaseNames(TestLoss)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(TestLoss(name, param={"config_file": config_file}))
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(
            TestLoss(test_method_name, param={"config_file": config_file}))
        unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

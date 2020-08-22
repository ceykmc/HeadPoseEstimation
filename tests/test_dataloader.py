import argparse
import unittest

import datasets  # noqa: F401,F403
from mmcv import Config
from mmcls.datasets import build_dataloader, build_dataset


def argument_parser():
    parser = argparse.ArgumentParser(description="anchors test")
    parser.add_argument("--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--test-method", default="", help="test method name")
    args = parser.parse_args()
    return args


class TestDataloader(unittest.TestCase):
    def __init__(self, methodName="runTest", param=None):
        super().__init__(methodName)
        assert "config_file" in param
        self.cfg = Config.fromfile(param["config_file"])

    def test_create_dataloader(self):
        train_dataset = build_dataset(self.cfg.data.train)
        print(F"train sample number: {len(train_dataset)}")
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            num_gpus=2,
            dist=False,
            shuffle=False)
        for index, data in enumerate(train_dataloader):
            for k, v in data.items():
                print(k, type(v.data[0]))
                print(data["img"].data.shape)
                print(data["gt_labels"].data)
                print(data["gt_angles"].data)
            break


def main():
    args = argument_parser()
    config_file = args.config_file
    test_method_name = args.test_method

    if len(test_method_name) == 0:
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(TestDataloader)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(
                TestDataloader(name, param={"config_file": config_file}))
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(
            TestDataloader(test_method_name,
                           param={"config_file": config_file}))
        unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()

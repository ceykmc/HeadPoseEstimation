"""300W-LP Dataset
"""

from pathlib import Path

from mmcls.datasets import DATASETS, BaseDataset


@DATASETS.register_module()
class W300LP(BaseDataset):
    dataset_names = [
        "AFW", "AFW_Flip", "HELEN", "HELEN_Flip", "IBUG", "IBUG_Flip", "LFPW",
        "LFPW_Flip"
    ]

    def __init__(self, data_prefix, pipeline, ann_file=None, test_mode=False):
        super().__init__(data_prefix, pipeline, ann_file, test_mode)

    def load_annotations(self):
        data_infos = list()
        for dataset_name in self.dataset_names:
            dataset_folder = Path(self.data_prefix) / dataset_name
            for image_path in dataset_folder.glob("*.jpg"):
                info = dict()
                info["img_prefix"] = dataset_folder
                info["img_info"] = dict()
                info["img_info"]["filename"] = image_path.name
                data_infos.append(info)
        return data_infos

    def evaluate(self, results, logger=None):
        nums = list()
        yaw_dists, pitch_dists, roll_dists = list(), list(), list()
        for result in results:
            nums.append(result["num_samples"].cpu().item())
            yaw_dists.append(result["yaw_dist"].cpu().item())
            pitch_dists.append(result["pitch_dist"].cpu().item())
            roll_dists.append(result["roll_dist"].cpu().item())
        assert sum(nums) == len(self.data_infos)
        assert len(yaw_dists) > 0 and len(pitch_dists) > 0 and len(
            roll_dists) > 0
        eval_results = dict()
        eval_results["yaw_dists"] = sum(yaw_dists) / len(yaw_dists)
        eval_results["pitch_dists"] = sum(pitch_dists) / len(pitch_dists)
        eval_results["roll_dists"] = sum(roll_dists) / len(roll_dists)
        return eval_results

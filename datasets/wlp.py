"""300W-LP Dataset
"""

from pathlib import Path
from mmcls.datasets import BaseDataset, DATASETS


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

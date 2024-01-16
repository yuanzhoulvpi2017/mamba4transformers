import os
from dataclasses import dataclass, field
from typing import Any, List, Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from pathlib import Path
from typing import Optional
import os
from datasets import load_dataset
import logging
from .arguments import DataArguments


logger = logging.getLogger(__name__)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for root, dir, file_name in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(
    data_file_list: List[str], cache_dir: Optional[str] = "cache_data"
) -> Dataset:
    all_file_list = data_file_list
    data_files = {"train": all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )["train"]
    return raw_datasets


class PreTrainDataset(Dataset):
    def __init__(self, dataargs: DataArguments) -> None:
        super().__init__()
        self.dataargs = dataargs
        self.data_dir = dataargs.data_dir

        self.alldataset = self.generate_data()

    def generate_data(self) -> Dataset:
        datafile_index_file = get_all_datapath(self.data_dir)
        data_file_list = [i for i in datafile_index_file if "index.json" not in i]
        alldataset = load_dataset_from_path(
            data_file_list, cache_dir=self.dataargs.cache_local_dir
        )
        return alldataset

    def __len__(self):
        return len(self.alldataset)

    def __getitem__(self, index) -> Any:
        return self.alldataset[index]


@dataclass
class PretrainDataCollator:
    max_length: int = field(default=1024)
    tokenizer: PreTrainedTokenizer = field(default=None)

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        features = [
            i["input_ids"][: (self.max_length - 1)] + [self.tokenizer.eos_token_id]
            for i in features
        ]
        feature_length = [len(i) for i in features]
        max_features_item_length = max(feature_length)
        features_tenosr = torch.stack(
            [
                torch.tensor(
                    value
                    + [self.tokenizer.eos_token_id]
                    * (max_features_item_length - feature_length[index]),
                    dtype=torch.long,
                )
                for index, value in enumerate(features)
            ]
        )
        return {"input_ids": features_tenosr}


if __name__ == "__main__":
    a = PreTrainDataset(data_dir="data/data4pretrain")
    a[0]

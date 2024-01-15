import math
import os
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from pathlib import Path
from typing import Optional
import os
from datasets import load_dataset
from itertools import islice


class PreTrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.generator = self.get_item()

    def get_item(self):
        for i in [[1, 2, 3], [4, 5, 6]]:
            for j in i:
                yield j

    def __len__(self):
        return 6

    def __getitem__(self, index) -> Any:
        return islice(self.generator, index, index + 1).__next__()

    # def __getitem__(self, index) -> Any:
    #     if index <= self.__len__() - 1:
    #         return next(self.generator)
    #     else:
    #         raise IndexError(f"index must less than {self.__len__() - 1}")


if __name__ == "__main__":
    a = PreTrainDataset()
    a[0]

import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class DataArguments:
    data_dir: str = field(default=None)
    cache_local_dir: str = field(default=None)

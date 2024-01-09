from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Union
import math

logger = logging.get_logger(__name__)


class MambaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "mamba"

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        d_conv: int = 4,
        pad_vocab_size_multiple: int = 8,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.expand = expand
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_conv = d_conv
        self.d_inner = int(self.expand * self.d_model)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )

import logging

from train.arguments import DataArguments, ModelArguments
from train.data import PretrainDataCollator, PreTrainDataset
from train.trainer import PreTrainTrainer
import sys

from mamba.configuration_mamba import MambaConfig
from mamba.modeling_mamba import MambaForCausalLM

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    PreTrainedTokenizer,
    HfArgumentParser,
)

logger = logging.getLogger(__name__)


def create_model_tokenizer(
    modelargs: ModelArguments,
) -> tuple[MambaForCausalLM, PreTrainedTokenizer | PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        modelargs.tokenizer_path, trust_remote_code=True
    )

    modelconfig = MambaConfig(
        vocab_size=len(tokenizer),
        d_model=1024,
        n_layer=4,
        d_state=16,
    )
    model = MambaForCausalLM(config=modelconfig).cuda()

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(
        f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    return model, tokenizer


def load_dataset(dataargs: DataArguments):
    dataset = PreTrainDataset(dataargs)
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    model, tokenizer = create_model_tokenizer(model_args)
    train_dataset = load_dataset(data_args)

    data_collator = PretrainDataCollator(
        max_length=data_args.max_seq_length, tokenizer=tokenizer
    )

    trainer = PreTrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()

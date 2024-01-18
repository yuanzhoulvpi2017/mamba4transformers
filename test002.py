from train.data import PreTrainDataset, PretrainDataCollator
from train.arguments import DataArguments
from transformers import AutoTokenizer

def main1():
    dataargument = DataArguments(
        data_dir="data/data4pretrain", cache_local_dir="cache_data2"
    )
    pt = PreTrainDataset(dataargs=dataargument)
    tokenizer = AutoTokenizer.from_pretrained("internlm_tokenizer", trust_remote_code=True)


    dc = PretrainDataCollator(max_length=1024, tokenizer=tokenizer)
    pt[0]

    a = dc([pt[i] for i in range(20)])


main1()

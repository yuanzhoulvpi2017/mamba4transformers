from transformers import TrainingArguments

from transformers import Trainer


class PreTrainTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"].clone().cuda()
        modelout = model(input_ids=inputs["input_ids"].cuda(), labels=labels)
        loss = modelout.loss
        return loss

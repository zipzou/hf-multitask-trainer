import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments

from hf_mtask_trainer import HfMultiTaskTrainer


class TestModel(nn.Module):

    def __init__(self, ) -> None:
        super().__init__()
        self.scaler = nn.Parameter(torch.ones(1))

    def forward(self, x):
        test_tensor = x + self.scaler
        test_np = np.array(np.random.randn()).astype(np.float32)
        test_int = random.randint(1, 100)
        test_float = random.random()

        self.report_metrics(
            tensor=test_tensor,
            np=test_np,
            integer=test_int,
            fp_num=test_float
        )

        loss = ((
            test_tensor + torch.from_numpy(test_np) + torch.tensor(test_int) +
            torch.tensor(test_float) - 0
        )).mean()

        outputs = (loss, )

        return outputs


class MockDataset(Dataset):

    def __len__(self):
        return 1000

    def __getitem__(self, index: int):
        return dict(x=torch.randn(10, dtype=torch.float32))


def main():
    parser = HfArgumentParser(TrainingArguments)
    args, = parser.parse_args_into_dataclasses()
    model = TestModel()
    ds = MockDataset()
    trainer = HfMultiTaskTrainer(model, args, train_dataset=ds)

    trainer.train()


if __name__ == '__main__':
    main()

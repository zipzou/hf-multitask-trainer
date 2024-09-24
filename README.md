# Enhanced Multitask Trainer for Separately Reporting Task's Metrics or Losses in HuggingFace Transformers

The [HuggingFace transformers](https://github.com/huggingface/transformers) library is widely used for model training. For example, to adapt a pretrained BERT model to a specific task domain, we often continue pretraining the model with two tasks in BERT: 1) Next Sentence Prediction (NSP) using the `[CLS]` token, and 2) Masked Language Modeling (MLM) using masked tokens. 

**A key issue is that the default `Trainer` in `transformers` assumes the first element of the output is the final loss to minimize. The loss returned by the `forward` method must be a scalar, so when training a multitask model like BERT, the loss needs to be combined.**

**The `Trainer` class offers command-line arguments to control the training process. However, it only provides a combined loss value for all tasks, which obscures the individual losses of each task. This makes it challenging to monitor training and debug different task settings. Additionally, the `Tensorboard` report only shows the combined loss in its metrics.**

To facilitate multitask model training and review the loss of each task, as well as other training metrics, this trainer implementation is simple and useful.

The trainer works like the original `Trainer` in the `transformers` library. You just need to call the `report_metrics(...)` method to report the metrics that are important to you.

By the way, another utility you might need is [parser-binding](https://github.com/zipzou/parser-binding), which builds argument parsers from dataclasses and reads the arguments from command line scripts.

## Usage

Follow these steps to use the `HfMultiTaskTrainer`:

1. Install the trainer:

    ```sh
    pip install hf-mtask-trainer
    ```

2. Replace the default trainer with `HfMultiTaskTrainer`:

    ```python
    from hf_mtask_trainer import HfMultiTaskTrainer

    class Trainer(HfMultiTaskTrainer):
        def __init__(...):
            super().__init__(...)
            # Additional initialization code
    ```

    Alternatively, you can directly instantiate the `HfMultiTaskTrainer`:

    ```python
    trainer = HfMultiTaskTrainer(...)
    ```

3. Report metrics in the model:

    ```python
    import torch.nn as nn

    class Model(nn.Module):

        supports_report_metrics: bool = True

        def __init__(...):
            super().__init__(...)
            # Additional initialization code
        
        def forward(self, inputs, ...):
            # Calculate metrics like loss, accuracy, etc.
            task1_loss = ...
            task2_loss = ...
            acc = ...
            f1 = ...
            # Report the metrics
            self.report_metrics(loss1=task1_loss, loss2=task2_loss, acc=acc, f1=f1)
    ```

    Add a flag `supports_report_metrics` where you need to report metrics, otherwise, the `report_metrics` would be not accessible.

4. Start training the model:

    As usual, call `trainer.train()` to start training.

Now you can enjoy multitask training. If you set `--report tensorboard`, the metrics reported in the model will be displayed in Tensorboard diagrams.


## Demo

We give a simple demo to mock a multi-task training in [test_trainer.py](./test_trainer.py).

The source code is:
```python
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments

from hf_mtask_trainer import HfMultiTaskTrainer

# The model class
class TestModel(nn.Module):
    supports_report_metrics: bool = True # IMPORTANT

    def __init__(self, ) -> None:
        super().__init__()
        self.scaler = nn.Parameter(torch.ones(1))

    def forward(self, x):
        test_tensor = x + self.scaler
        test_np = np.array(np.random.randn()).astype(np.float32)
        test_int = random.randint(1, 100)
        test_float = random.random()
        if hasattr(self, report_metrics): # checking if the report method is accessible or not is the robust practice
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

# Mock dataset
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
    # Use HfMultiTaskTrainer rather than Trainer
    trainer = HfMultiTaskTrainer(model, args, train_dataset=ds)

    trainer.train()


if __name__ == '__main__':
    main()

```

Run the script to start training: `python test_trainer.py --output_dir ./test-output --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --logging_steps 10 --num_train_epochs 10`.

The progress in the terminal is:
```sh
{'loss': 55.509, 'grad_norm': 1.0, 'learning_rate': 4.8387096774193554e-05, 'tensor': 0.9841784507036209, 'np': -0.21863683552946894, 'integer': 54.3, 'fp_num': 0.4434714410935673, 'epoch': 0.32}               
{'loss': 48.2661, 'grad_norm': 1.0, 'learning_rate': 4.67741935483871e-05, 'tensor': 0.9985833063721656, 'np': -0.02904345905408263, 'integer': 46.725, 'fp_num': 0.5715085473120125, 'epoch': 0.64}              
{'loss': 46.4612, 'grad_norm': 1.0, 'learning_rate': 4.516129032258064e-05, 'tensor': 0.9966234847903251, 'np': 0.010173140384722501, 'integer': 44.95, 'fp_num': 0.5043715258219943, 'epoch': 0.96}              
{'loss': 48.6079, 'grad_norm': 1.0, 'learning_rate': 4.3548387096774194e-05, 'tensor': 0.9955430060625077, 'np': -0.03289987128227949, 'integer': 47.175, 'fp_num': 0.47028585139293366, 'epoch': 1.28}           
{'loss': 50.091, 'grad_norm': 1.0, 'learning_rate': 4.1935483870967746e-05, 'tensor': 0.9734495922923088, 'np': 0.06655221048276871, 'integer': 48.55, 'fp_num': 0.5009696474466848, 'epoch': 1.6}                
{'loss': 52.1638, 'grad_norm': 1.0, 'learning_rate': 4.032258064516129e-05, 'tensor': 1.0023577958345413, 'np': 0.18944044597446918, 'integer': 50.5, 'fp_num': 0.47205086657725437, 'epoch': 1.92}               
{'loss': 61.3063, 'grad_norm': 1.0, 'learning_rate': 3.870967741935484e-05, 'tensor': 1.0168104887008667, 'np': -0.10900555825792253, 'integer': 60.0, 'fp_num': 0.39849607236524115, 'epoch': 2.24}              
{'loss': 55.318, 'grad_norm': 1.0, 'learning_rate': 3.7096774193548386e-05, 'tensor': 1.015606315433979, 'np': 0.21950888196006418, 'integer': 53.575, 'fp_num': 0.5078790131376146, 'epoch': 2.56}               
{'loss': 57.1703, 'grad_norm': 1.0, 'learning_rate': 3.548387096774194e-05, 'tensor': 1.0161942049860955, 'np': -0.08120755353011191, 'integer': 55.675, 'fp_num': 0.5603439507938002, 'epoch': 2.88}             
{'loss': 47.6687, 'grad_norm': 1.0, 'learning_rate': 3.387096774193548e-05, 'tensor': 0.9780291050672532, 'np': 0.21060471932869404, 'integer': 46.025, 'fp_num': 0.4550899259063651, 'epoch': 3.2}               
{'loss': 50.6742, 'grad_norm': 1.0, 'learning_rate': 3.2258064516129034e-05, 'tensor': 0.9773322150111199, 'np': 0.053728557180147615, 'integer': 49.15, 'fp_num': 0.4931880990797102, 'epoch': 3.52}             
{'loss': 55.3104, 'grad_norm': 1.0, 'learning_rate': 3.0645161290322585e-05, 'tensor': 0.962137694656849, 'np': -0.079732296615839, 'integer': 53.975, 'fp_num': 0.45303205544101893, 'epoch': 3.84}              
{'loss': 55.3539, 'grad_norm': 1.0, 'learning_rate': 2.9032258064516133e-05, 'tensor': 1.0214665666222573, 'np': -0.15776186664588748, 'integer': 53.9, 'fp_num': 0.590140296440284, 'epoch': 4.16}               
{'loss': 49.332, 'grad_norm': 1.0, 'learning_rate': 2.7419354838709678e-05, 'tensor': 1.0191335454583168, 'np': -0.2712035422213376, 'integer': 48.025, 'fp_num': 0.5590896723075907, 'epoch': 4.48}              
{'loss': 49.8865, 'grad_norm': 1.0, 'learning_rate': 2.5806451612903226e-05, 'tensor': 1.0170967370271682, 'np': 0.02669397685676813, 'integer': 48.275, 'fp_num': 0.5677363725430722, 'epoch': 4.8}              
{'loss': 55.0644, 'grad_norm': 1.0, 'learning_rate': 2.4193548387096777e-05, 'tensor': 0.99910968542099, 'np': 0.12097712438553572, 'integer': 53.475, 'fp_num': 0.4693036682925622, 'epoch': 5.12}               
{'loss': 56.9469, 'grad_norm': 1.0, 'learning_rate': 2.258064516129032e-05, 'tensor': 1.0159066557884215, 'np': -0.06122639870736748, 'integer': 55.6, 'fp_num': 0.3922143274213026, 'epoch': 5.44}               
{'loss': 58.3238, 'grad_norm': 1.0, 'learning_rate': 2.0967741935483873e-05, 'tensor': 0.9946490600705147, 'np': -0.038768217992037536, 'integer': 56.875, 'fp_num': 0.49290766579450906, 'epoch': 5.76}          
{'loss': 57.8349, 'grad_norm': 1.0, 'learning_rate': 1.935483870967742e-05, 'tensor': 0.9948656186461449, 'np': -0.15342782847583294, 'integer': 56.55, 'fp_num': 0.4434852700815277, 'epoch': 6.08}              
{'loss': 57.5093, 'grad_norm': 1.0, 'learning_rate': 1.774193548387097e-05, 'tensor': 0.9814934283494949, 'np': 0.17727854922413827, 'integer': 55.85, 'fp_num': 0.5005189062297719, 'epoch': 6.4}                
{'loss': 54.0808, 'grad_norm': 1.0, 'learning_rate': 1.6129032258064517e-05, 'tensor': 1.0003552585840225, 'np': 0.09905800204724073, 'integer': 52.425, 'fp_num': 0.5563636991813741, 'epoch': 6.72}             
{'loss': 41.9312, 'grad_norm': 1.0, 'learning_rate': 1.4516129032258066e-05, 'tensor': 0.9884074732661248, 'np': 0.1483861011918634, 'integer': 40.275, 'fp_num': 0.51941084196083, 'epoch': 7.04}                
{'loss': 54.1181, 'grad_norm': 1.0, 'learning_rate': 1.2903225806451613e-05, 'tensor': 1.0151973858475685, 'np': 0.47866107723675666, 'integer': 52.175, 'fp_num': 0.4492639089144623, 'epoch': 7.36}             
{'loss': 50.6587, 'grad_norm': 1.0, 'learning_rate': 1.129032258064516e-05, 'tensor': 0.9820004492998123, 'np': -0.012274338398128748, 'integer': 49.2, 'fp_num': 0.4889366814531261, 'epoch': 7.68}              
{'loss': 55.0801, 'grad_norm': 1.0, 'learning_rate': 9.67741935483871e-06, 'tensor': 0.9795809179544449, 'np': -0.07257360897492618, 'integer': 53.725, 'fp_num': 0.448120297698113, 'epoch': 8.0}                
{'loss': 44.1352, 'grad_norm': 1.0, 'learning_rate': 8.064516129032258e-06, 'tensor': 0.9734664395451545, 'np': 0.286221909429878, 'integer': 42.375, 'fp_num': 0.5005484995389671, 'epoch': 8.32}                
{'loss': 66.0453, 'grad_norm': 1.0, 'learning_rate': 6.451612903225806e-06, 'tensor': 0.9795126229524612, 'np': 0.030494442163035273, 'integer': 64.525, 'fp_num': 0.5103026267522799, 'epoch': 8.64}             
{'loss': 56.4957, 'grad_norm': 1.0, 'learning_rate': 4.838709677419355e-06, 'tensor': 0.9856317490339279, 'np': 0.2455663602799177, 'integer': 54.775, 'fp_num': 0.48952095056963413, 'epoch': 8.96}              
{'loss': 58.896, 'grad_norm': 1.0, 'learning_rate': 3.225806451612903e-06, 'tensor': 0.9927483782172203, 'np': 0.14382120433729143, 'integer': 57.275, 'fp_num': 0.4843773558505675, 'epoch': 9.28}               
{'loss': 51.3854, 'grad_norm': 1.0, 'learning_rate': 1.6129032258064516e-06, 'tensor': 0.974456462264061, 'np': -0.03793883747421205, 'integer': 49.975, 'fp_num': 0.4738723365026173, 'epoch': 9.6}              
{'loss': 51.1056, 'grad_norm': 1.0, 'learning_rate': 0.0, 'tensor': 1.0064959138631822, 'np': 0.17969400193542243, 'integer': 49.375, 'fp_num': 0.5444181010304485, 'epoch': 9.92}                                
{'train_runtime': 0.4796, 'train_samples_per_second': 20852.484, 'train_steps_per_second': 646.427, 'train_loss': 53.31389662219632, 'epoch': 9.92}                                                               
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 310/310 [00:00<00:00, 648.81it/s]
```

## Limitation

This trainer has not been fully tested yet but works for simple multitask training. Please report any issues if this plugin does not work for you.


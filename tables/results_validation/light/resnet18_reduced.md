## Experiment Results outputs/OrganoidsINRIA_reduced/resnet18

| Run | Model | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | resnet18 | yes | CE | 100 | 0.1 | AdamW | warmup_cosine | no | stratified | 66 | 52 |
| 02 | resnet18 | yes | CE | 100 | 0.05 | AdamW | warmup_cosine | no | stratified | 96 | 52 |
| 03 | resnet18 | yes | CE | 100 | 0.01 | AdamW | warmup_cosine | no | stratified | 66 | 38 |
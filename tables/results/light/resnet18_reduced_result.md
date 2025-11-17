## Experiment Results outputs/OrganoidsINRIA_reduced/resnet18

| Run | Model | Aug | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | resnet18 | yes | no | CE | 132/150 | 0.05 | AdamW | warmup_cosine | 94 | 92 | 92 |
| 02 | resnet18 | yes | yes | CE | 106/150 | 0.05 | AdamW | warmup_cosine | 93 | 88 | 89 |
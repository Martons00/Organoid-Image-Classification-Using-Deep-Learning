## Experiment Results outputs/OrganoidsINRIA_reduced/densenet

| Run | Model | Aug | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | densenet | no | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 02 | densenet | no | yes | CE | 92/150 | 0.05 | AdamW | warmup_cosine | 99 | 75 | 79 |
| 01 | resnet18 | yes | no | CE | 132/150 | 0.05 | AdamW | warmup_cosine | 94 | 92 | 92 |
| 02 | resnet18 | yes | yes | CE | 106/150 | 0.05 | AdamW | warmup_cosine | 93 | 88 | 89 |
| 01 | resnet50 | yes | no | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 80 | 60 | 65 |
| 02 | resnet50 | yes | yes | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 78 | 78 | 74 |
| 01 | swinunetr | yes | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 02 | swinunetr | yes | yes | CE | 142*/150 | 0.0006 | AdamW | warmup_cosine | 86 | 87 | 86 |
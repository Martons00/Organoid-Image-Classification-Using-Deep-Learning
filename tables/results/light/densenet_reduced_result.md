## Experiment Results outputs/OrganoidsINRIA_reduced/densenet

| Run | Model | Aug | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | densenet | no | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 02 | densenet | no | yes | CE | 92/150 | 0.05 | AdamW | warmup_cosine | 99 | 75 | 79 |
| 03 | densenet | yes | no | CE | 150/150 | 0.05 | AdamW | warmup_cosine | 60 | 89 | 86 |
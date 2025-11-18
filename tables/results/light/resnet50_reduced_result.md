## Experiment Results outputs/OrganoidsINRIA_reduced/resnet50

| Run | Model | Aug | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | resnet50 | yes | no | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 80 | 60 | 65 |
| 02 | resnet50 | yes | yes | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 78 | 78 | 74 |
| 03 | resnet50 | yes | no | CE | 90/150 | 0.001 | AdamW | warmup_cosine | 82 | 66 | 59 |
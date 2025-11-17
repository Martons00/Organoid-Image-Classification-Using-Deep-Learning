## Experiment Results outputs/OrganoidsINRIA

| Run | Model | Aug | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinunetr | yes | no | CE | 118/150 | 0.0006 | AdamW | warmup_cosine | 85 | 82 | 90 |
| 01 | resnet18 | yes | no | CE | 150*/150 | 0.05 | AdamW | warmup_cosine | 99 | 63 | 72 |
| 01 | densenet | no | no | CE | 88/150 | 0.05 | AdamW | warmup_cosine | 47 | 50 | 53 |
| 01 | resnet50 | yes | no | CE | 84/150 | 0.0005 | AdamW | warmup_cosine | 50 | 50 | 52 |

## Experiment Results outputs/OrganoidsINRIA_reduced/swinunetr

| Run | Model | Aug | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinunetr | yes | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 02 | swinunetr | yes | yes | CE | 142*/150 | 0.0006 | AdamW | warmup_cosine | 86 | 87 | 86 |
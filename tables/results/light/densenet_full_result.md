## Experiment Results outputs/OrganoidsINRIA/densenet

| Run | Model | Aug | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | densenet | no | no | CE | 88/150 | 0.05 | AdamW | warmup_cosine | 47 | 50 | 53 |
| 02 | densenet | yes | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 50 | 50 | 53 |
## Experiment Results outputs/OrganoidsINRIA/densenet

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_H | densenet | no | 128x128x128 | no | CE | 88/150 | 0.05 | AdamW | warmup_cosine | 47 | 50 | 53 |
| 02_H | densenet | yes | 128x128x128 | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 50 | 50 | 53 |
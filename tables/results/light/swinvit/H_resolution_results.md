## Experiment Results outputs/OrganoidsINRIA/swinvit

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_H | swinvit | yes | 128x128x128 | no | CE | 164/300 | 0.006 | AdamW | warmup_cosine | 82 | 80 | 85 |
| 02_H | swinvit | yes | 128x128x128 | no | CE | 196/300 | 0.04 | AdamW | warmup_cosine | 73 | 74 | 79 |
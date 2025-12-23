## Experiment Results outputs/OrganoidsINRIA_reduced_128/swinvit

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.05 | AdamW | warmup_cosine | 39 | 49 | 49 |
| 02_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.5 | AdamW | warmup_cosine | 38 | 49 | 49 |
| 03_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.005 | AdamW | warmup_cosine | 86 | 83 | 83 |
| 04_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.001 | AdamW | warmup_cosine | 86 | 85 | 84 |
| 05_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.01 | AdamW | warmup_cosine | 44 | 49 | 49 |
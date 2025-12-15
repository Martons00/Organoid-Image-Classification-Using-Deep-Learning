## Experiment Results outputs/OrganoidsINRIA_reduced/swinvit

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 81 | 83 | 84 |
| 02 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.006 | AdamW | warmup_cosine | 84 | 88 | 87 |
| 03 | swinvit | yes | 128x128x128 | no | CE | 218/300 | 0.01 | AdamW | warmup_cosine | 80 | 83 | 80 |
| 04_scratch | swinvit | yes | 64x64x64 | no | CE | 82/300 | 0.006 | AdamW | warmup_cosine | 45 | 48 | 48 |
| 05_64 | swinvit | yes | 64x64x64 | no | CE | 82/300 | 0.006 | AdamW | warmup_cosine | 42 | 49 | 49 |
| 06_64 | swinvit | yes | 64x64x64 | no | CE | 224/300 | 0.003 | AdamW | warmup_cosine | 80 | 83 | 87 |
| 07_scratch | swinvit | yes | 128x128x128 | no | CE | 84/300 | 0.005 | AdamW | warmup_cosine | 50 | 49 | 49 |
| 08 | swinvit | yes | 128x128x128 | no | CE | */300 | 0.05 | AdamW | warmup_cosine | 56 | 75 | 71 |
| 09_scratch | swinvit | yes | 128x128x128 | no | CE | */300 | 0.0005 | AdamW | warmup_cosine | 45 | 83 | 83 |
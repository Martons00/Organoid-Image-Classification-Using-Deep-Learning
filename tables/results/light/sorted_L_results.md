## Full Results

| Rank | Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 01_L | resnet18 | yes | 128x128x128 | no | CE | */150 | 0.05 | AdamW | warmup_cosine | 97 | 93 | 94 |
| 2 | 01_L | densenet | yes | 128x128x128 | no | CE | */150 | 0.01 | AdamW | warmup_cosine | 64 | 91 | 89 |
| 3 | 01_L | swinunetr | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 87 | 83 | 88 |
| 4 | 02_L | swinunetr | yes | 128x128x128 | no | CE | */150 | 0.005 | AdamW | warmup_cosine | 82 | 81 | 85 |
| 5 | 03_L | swinunetr | yes | 128x128x128 | no | CE | */300 | 0.005 | AdamW | warmup_cosine | 91 | 85 | 84 |
| 6 | 04_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.001 | AdamW | warmup_cosine | 86 | 85 | 84 |
| 7 | 03_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.005 | AdamW | warmup_cosine | 86 | 83 | 83 |
| 8 | 01_L | swinunetr+noah | yes | 128x128x128 | no | CE | */150 | 0.0006 | AdamW | warmup_cosine | 82 | 84 | 79 |
| 9 | 01_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.05 | AdamW | warmup_cosine | 39 | 49 | 49 |
| 10 | 02_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.5 | AdamW | warmup_cosine | 38 | 49 | 49 |
| 11 | 05_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.01 | AdamW | warmup_cosine | 44 | 49 | 49 |

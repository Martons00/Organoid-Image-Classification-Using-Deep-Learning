## Full Results

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_L | densenet | yes | 128x128x128 | no | CE | */150 | 0.01 | AdamW | warmup_cosine | 64 | 91 | 89 |
| 01_L | resnet18 | yes | 128x128x128 | no | CE | */150 | 0.05 | AdamW | warmup_cosine | 97 | 93 | 94 |
| 01_L | swinunetr | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 87 | 83 | 88 |
| 01_L | swinunetr+noah | yes | 128x128x128 | no | CE | */150 | 0.0006 | AdamW | warmup_cosine | 82 | 84 | 79 |
| 01_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.05 | AdamW | warmup_cosine | 39 | 49 | 49 |
| 02_L | swinvit | yes | 128x128x128 | no | CE | */300 | 0.5 | AdamW | warmup_cosine | 38 | 49 | 49 |

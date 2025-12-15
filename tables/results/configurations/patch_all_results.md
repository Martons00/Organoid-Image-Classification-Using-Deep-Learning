## Full Results

| Rank | Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 03 | resnet18 | yes | 128x128x128 | no | CE | 128/150 | 0.08 | AdamW | warmup_cosine | 94 | 89 | 94 |
| 2 | 01 | resnet18 | yes | 128x128x128 | no | CE | 132/150 | 0.05 | AdamW | warmup_cosine | 94 | 92 | 92 |
| 3 | 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 5 | 06_64 | swinunetr | yes | 64x64x64 | no | CE | 150/150 | 0.001 | AdamW | warmup_cosine | 84 | 88 | 90 |
| 8 | 01 | densenet | no | 128x128x128 | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 10 | 03 | swinunetr | yes | 64x64x64 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 89 | 90 | 88 |
| 11 | 02 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.006 | AdamW | warmup_cosine | 84 | 88 | 87 |
| 12 | 06_64 | swinvit | yes | 64x64x64 | no | CE | 224/300 | 0.003 | AdamW | warmup_cosine | 80 | 83 | 87 |
| 13 | 03 | densenet | yes | 128x128x128 | no | CE | 150/150 | 0.05 | AdamW | warmup_cosine | 60 | 89 | 86 |
| 14 | 07_64 | densenet | yes | 64x64x64 | no | CE | 97/150 | 0.05 | AdamW | warmup_cosine | 50 | 85 | 86 |
| 21 | 01 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 81 | 83 | 84 |
| 23 | 05 | densenet | yes | 64x64x64 | no | CE | 110/150 | 0.005 | AdamW | warmup_cosine | 45 | 81 | 82 |
| 24 | 04 | densenet | yes | 64x64x64 | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 45 | 77 | 80 |
| 25 | 03 | swinvit | yes | 128x128x128 | no | CE | 218/300 | 0.01 | AdamW | warmup_cosine | 80 | 83 | 80 |
| 28 | 09_64 | resnet18 | yes | 64x64x64 | no | CE | 88/150 | 0.1 | AdamW | warmup_cosine | 90 | 72 | 78 |
| 29 | 04 | resnet18 | yes | 64x64x64 | no | CE | 84/150 | 0.08 | AdamW | warmup_cosine | 87 | 73 | 74 |
| 32 | 10_64 | resnet18 | yes | 64x64x64 | no | CE | 90/150 | 0.05 | AdamW | warmup_cosine | 94 | 73 | 68 |
| 33 | 08_64 | densenet | yes | 64x64x64 | no | CE | 90/150 | 0.01 | AdamW | warmup_cosine | 49 | 77 | 66 |
| 34 | 05 | resnet18 | yes | 64x64x64 | no | CE | 82/150 | 0.1 | AdamW | warmup_cosine | 60 | 51 | 65 |
| 36 | 06 | resnet18 | yes | 64x64x64 | no | CE | 92/150 | 0.01 | AdamW | warmup_cosine | 99 | 65 | 62 |
| 43 | 05_64 | swinvit | yes | 64x64x64 | no | CE | 82/300 | 0.006 | AdamW | warmup_cosine | 42 | 49 | 49 |
| 46 | 06 | densenet | yes | 64x64x64 | no | CE | 114/150 | 0.0005 | AdamW | warmup_cosine | 48 | 56 | 46 |

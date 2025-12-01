## Full Results

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03 | resnet18 | yes | 128x128x128 | no | CE | 128/150 | 0.08 | AdamW | warmup_cosine | 94 | 89 | 94 |
| 01 | resnet18 | yes | 128x128x128 | no | CE | 132/150 | 0.05 | AdamW | warmup_cosine | 94 | 92 | 92 |
| 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 01 | densenet | no | 128x128x128 | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 02 | resnet18 | yes | 128x128x128 | yes | CE | 106/150 | 0.05 | AdamW | warmup_cosine | 93 | 88 | 89 |
| 03 | swinunetr | yes | 64x64x64 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 89 | 90 | 88 |
| 02 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.006 | AdamW | warmup_cosine | 84 | 88 | 87 |
| 03 | densenet | yes | 128x128x128 | no | CE | 150/150 | 0.05 | AdamW | warmup_cosine | 60 | 89 | 86 |
| 02 | swinunetr | yes | 128x128x128 | yes | CE | 142*/150 | 0.0006 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 01 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0003 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 02 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0006 | AdamW | warmup_cosine | 85 | 83 | 84 |
| 01 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 81 | 83 | 84 |
| 07_scratch | resnet18 | yes | 128x128x128 | no | CE | 118/150 | 0.08 | AdamW | warmup_cosine | 45 | 83 | 83 |
| 05 | densenet | yes | 64x64x64 | no | CE | 110/150 | 0.005 | AdamW | warmup_cosine | 45 | 81 | 82 |
| 04 | densenet | yes | 64x64x64 | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 45 | 77 | 80 |
| 02 | densenet | no | 128x128x128 | yes | CE | 92/150 | 0.05 | AdamW | warmup_cosine | 99 | 75 | 79 |
| 04 | resnet18 | yes | 64x64x64 | no | CE | 84/150 | 0.08 | AdamW | warmup_cosine | 87 | 73 | 74 |
| 02 | resnet50 | yes | 128x128x128 | yes | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 78 | 78 | 74 |
| 05 | resnet18 | yes | 64x64x64 | no | CE | 82/150 | 0.1 | AdamW | warmup_cosine | 60 | 51 | 65 |
| 01 | resnet50 | yes | 128x128x128 | no | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 80 | 60 | 65 |
| 06 | resnet18 | yes | 64x64x64 | no | CE | 92/150 | 0.01 | AdamW | warmup_cosine | 99 | 65 | 62 |
| 03 | resnet50 | yes | 128x128x128 | no | CE | 90/150 | 0.001 | AdamW | warmup_cosine | 82 | 66 | 59 |
| 06 | densenet | yes | 64x64x64 | no | CE | 114/150 | 0.0005 | AdamW | warmup_cosine | 48 | 56 | 46 |

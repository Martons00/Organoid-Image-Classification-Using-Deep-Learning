## Full Results

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | densenet | no | 128x128x128 | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 02 | densenet | no | 128x128x128 | yes | CE | 92/150 | 0.05 | AdamW | warmup_cosine | 99 | 75 | 79 |
| 03 | densenet | yes | 128x128x128 | no | CE | 150/150 | 0.05 | AdamW | warmup_cosine | 60 | 89 | 86 |
| 04 | densenet | yes | 64x64x64 | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 45 | 77 | 80 |
| 05 | densenet | yes | 64x64x64 | no | CE | 110/150 | 0.005 | AdamW | warmup_cosine | 45 | 81 | 82 
| 01 | resnet18 | yes | 128x128x128 | no | CE | 132/150 | 0.05 | AdamW | warmup_cosine | 94 | 92 | 92 |
| 02 | resnet18 | yes | 128x128x128 | yes | CE | 106/150 | 0.05 | AdamW | warmup_cosine | 93 | 88 | 89 |
| 03 | resnet18 | yes | 128x128x128 | no | CE | 128/150 | 0.08 | AdamW | warmup_cosine | 94 | 89 | 94 |
| 04 | resnet18 | yes | 64x64x64 | no | CE | 84/150 | 0.08 | AdamW | warmup_cosine | 87 | 73 | 74 |
| 05 | resnet18 | yes | 64x64x64 | no | CE | 82/150 | 0.1 | AdamW | warmup_cosine | 60 | 51 | 65 |
| 06 | resnet18 | yes | 64x64x64 | no | CE | 92/150 | 0.01 | AdamW | warmup_cosine | 99 | 65 | 62 
| 01 | resnet50 | yes | 128x128x128 | no | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 80 | 60 | 65 |
| 02 | resnet50 | yes | 128x128x128 | yes | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 78 | 78 | 74 |
| 03 | resnet50 | yes | 128x128x128 | no | CE | 90/150 | 0.001 | AdamW | warmup_cosine | 82 | 66 | 59 
| 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 02 | swinunetr | yes | 128x128x128 | yes | CE | 142*/150 | 0.0006 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 03 | swinunetr | yes | 64x64x64 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 89 | 90 | 88 
| 01 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0003 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 02 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0006 | AdamW | warmup_cosine | 85 | 83 | 84 
| 01 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 81 | 83 | 84 |
| 02 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.006 | AdamW | warmup_cosine | 84 | 88 | 87 

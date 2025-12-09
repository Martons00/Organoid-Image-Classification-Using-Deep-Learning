## Full Results

| Rank | Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 03 | resnet18 | yes | 128x128x128 | no | CE | 128/150 | 0.08 | AdamW | warmup_cosine | 94 | 89 | 94 |
| 3 | 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 6 | 08_F | swinvit | yes | 128x128x128 | no | CE | 98/300 | 0.05 | AdamW | warmup_cosine | 100 | 90 | 90 |
| 7 | 01_F | swinunetr | yes | 128x128x128 | no | CE | 118/150 | 0.0006 | AdamW | warmup_cosine | 85 | 82 | 90 |
| 8 | 01 | densenet | no | 128x128x128 | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 16 | 01 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0003 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 17 | 01_F | swinunetr+noah | yes | 128x128x128 | no | CE | 106/150 | 0.0006 | AdamW | warmup_cosine | 75 | 78 | 85 |
| 18 | 01_F | swin vit | yes | 128x128x128 | no | CE | 164/300 | 0.006 | AdamW | warmup_cosine | 82 | 80 | 85 |
| 21 | 01 | swinvit | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 81 | 83 | 84 |
| 30 | 02 | resnet50 | yes | 128x128x128 | yes | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 78 | 78 | 74 |
| 31 | 01_F | resnet18 | yes | 128x128x128 | no | CE | 150*/150 | 0.05 | AdamW | warmup_cosine | 99 | 63 | 72 |
| 40 | 01_F | densenet | no | 128x128x128 | no | CE | 88/150 | 0.05 | AdamW | warmup_cosine | 47 | 50 | 53 |
| 42 | 01_F | resnet50 | yes | 128x128x128 | no | CE | 84/150 | 0.0005 | AdamW | warmup_cosine | 50 | 50 | 52 |

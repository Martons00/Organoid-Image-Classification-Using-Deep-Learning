## Full Results

| Rank | Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 03 | resnet18 | yes | 128x128x128 | no | CE | 128/150 | 0.08 | AdamW | warmup_cosine | 94 | 89 | 94 |
| 3 | 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 4 | 05_scratch | swinunetr | yes | 128x128x128 | no | CE | 150/150 | 0.0001 | AdamW | warmup_cosine | 100 | 90 | 90 |
| 6 | 08 | swinvit | yes | 128x128x128 | no | CE | 98/300 | 0.05 | AdamW | warmup_cosine | 100 | 90 | 90 |
| 7 | 01 | densenet | no | 128x128x128 | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 16 | 04_scratch | swinunetr | yes | 128x128x128 | no | CE | 84/150 | 0.0006 | AdamW | warmup_cosine | 49 | 84 | 84 |
| 19 | 07_scratch | resnet18 | yes | 128x128x128 | no | CE | 118/150 | 0.08 | AdamW | warmup_cosine | 45 | 83 | 83 |
| 32 | 08_scratch | resnet18 | yes | 128x128x128 | no | CE | 90/300 | 0.001 | AdamW | warmup_cosine | 99 | 65 | 60 |
| 36 | 07_scratch | swinvit | yes | 128x128x128 | no | CE | 84/300 | 0.005 | AdamW | warmup_cosine | 50 | 49 | 49 |
| 37 | 04_scratch | swinvit | yes | 64x64x64 | no | CE | 82/300 | 0.006 | AdamW | warmup_cosine | 45 | 48 | 48 |

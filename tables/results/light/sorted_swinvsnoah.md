## Experiment Results outputs/OrganoidsINRIA_reduced/swinunetr+noah

| Rank | Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 2 | 05_scratch | swinunetr | yes | 128x128x128 | no | CE | 150/150 | 0.0001 | AdamW | warmup_cosine | 100 | 90 | 90 |
| 3 | 06_64 | swinunetr | yes | 64x64x64 | no | CE | 150/150 | 0.001 | AdamW | warmup_cosine | 84 | 88 | 90 |
| 4 | 01_H | swinunetr | yes | 128x128x128 | no | CE | 118/150 | 0.0006 | AdamW | warmup_cosine | 85 | 82 | 90 |
| 5 | 03 | swinunetr | yes | 64x64x64 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 89 | 90 | 88 |
| 6 | 01_L | swinunetr | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 87 | 83 | 88 |
| 7 | 01 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0003 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 8 | 02 | swinunetr | yes | 128x128x128 | yes | CE | 142*/150 | 0.0006 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 9 | 01_H | swinunetr+noah | yes | 128x128x128 | no | CE | 106/150 | 0.0006 | AdamW | warmup_cosine | 75 | 78 | 85 |
| 10 | 02_L | swinunetr | yes | 128x128x128 | no | CE | */150 | 0.005 | AdamW | warmup_cosine | 82 | 81 | 85 |
| 11 | 02 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0006 | AdamW | warmup_cosine | 85 | 83 | 84 |
| 12 | 04_scratch | swinunetr | yes | 128x128x128 | no | CE | 84/150 | 0.0006 | AdamW | warmup_cosine | 49 | 84 | 84 |
| 13 | 03_L | swinunetr | yes | 128x128x128 | no | CE | */300 | 0.005 | AdamW | warmup_cosine | 91 | 85 | 84 |
| 14 | 01_L | swinunetr+noah | yes | 128x128x128 | no | CE | */150 | 0.0006 | AdamW | warmup_cosine | 82 | 84 | 79 |
| 15 | 04 | swinunetr+noah | yes | 128x128x128 | no | CE | 88/150 | 0.005 | AdamW | warmup_cosine | 50 | 68 | 53 |

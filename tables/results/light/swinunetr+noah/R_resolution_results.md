## Experiment Results outputs/OrganoidsINRIA_reduced/swinunetr+noah

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0003 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 02 | swinunetr+noah | yes | 128x128x128 | no | CE | 150/150 | 0.0006 | AdamW | warmup_cosine | 85 | 83 | 84 |
| 03 | swinunetr+noah | yes | 128x128x128 | yes | CE | */150 | 0.0003 | AdamW | warmup_cosine | * | * | * |
| 04 | swinunetr+noah | yes | 128x128x128 | no | CE | */150 | 0.005 | AdamW | warmup_cosine | * | * | * |
| 05 | swinunetr+noah | yes | 128x128x128 | yes | CE | */150 | 0.0001 | AdamW | warmup_cosine | * | * | * |
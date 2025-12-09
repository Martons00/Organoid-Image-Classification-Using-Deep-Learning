## Experiment Results outputs/OrganoidsINRIA_reduced/swinunetr

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 02 | swinunetr | yes | 128x128x128 | yes | CE | 142*/150 | 0.0006 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 03 | swinunetr | yes | 64x64x64 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 89 | 90 | 88 |
| 04_scratch | swinunetr | yes | 128x128x128 | no | CE | 84/150 | 0.0006 | AdamW | warmup_cosine | 49 | 84 | 84 |
| 05_scratch | swinunetr | yes | 128x128x128 | no | CE | 150/150 | 0.0001 | AdamW | warmup_cosine | 100 | 90 | 90 |
| 06_64 | swinunetr | yes | 64x64x64 | no | CE | 150/150 | 0.001 | AdamW | warmup_cosine | 84 | 88 | 90 |
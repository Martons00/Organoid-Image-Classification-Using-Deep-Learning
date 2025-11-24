## Experiment Results outputs/OrganoidsINRIA_reduced/swinunetr

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinunetr | yes | 128x128x128 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 91 | 86 | 90 |
| 02 | swinunetr | yes | 128x128x128 | yes | CE | 142*/150 | 0.0006 | AdamW | warmup_cosine | 86 | 87 | 86 |
| 03 | swinunetr | yes | 64x64x64 | no | CE | 142/150 | 0.0006 | AdamW | warmup_cosine | 89 | 90 | 88 |
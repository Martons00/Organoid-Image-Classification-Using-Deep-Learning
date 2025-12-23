## Experiment Results outputs/OrganoidsINRIA_reduced_128/swinunetr

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_L | swinunetr | yes | 128x128x128 | no | CE | */150 | 0.001 | AdamW | warmup_cosine | 87 | 83 | 88 |
| 02_L | swinunetr | yes | 128x128x128 | no | CE | */150 | 0.005 | AdamW | warmup_cosine | 82 | 81 | 85 |
| 03_L | swinunetr | yes | 128x128x128 | no | CE | */300 | 0.005 | AdamW | warmup_cosine | 91 | 85 | 84 |
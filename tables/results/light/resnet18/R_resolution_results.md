## Experiment Results outputs/OrganoidsINRIA_reduced/resnet18

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | resnet18 | yes | 128x128x128 | no | CE | 132/150 | 0.05 | AdamW | warmup_cosine | 94 | 92 | 92 |
| 02 | resnet18 | yes | 128x128x128 | yes | CE | 106/150 | 0.05 | AdamW | warmup_cosine | 93 | 88 | 89 |
| 03 | resnet18 | yes | 128x128x128 | no | CE | 128/150 | 0.08 | AdamW | warmup_cosine | 94 | 89 | 94 |
| 04 | resnet18 | yes | 64x64x64 | no | CE | 84/150 | 0.08 | AdamW | warmup_cosine | 87 | 73 | 74 |
| 05 | resnet18 | yes | 64x64x64 | no | CE | 82/150 | 0.1 | AdamW | warmup_cosine | 60 | 51 | 65 |
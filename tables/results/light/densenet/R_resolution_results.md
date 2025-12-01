## Experiment Results outputs/OrganoidsINRIA_reduced/densenet

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | densenet | no | 128x128x128 | no | CE | 108/150 | 0.05 | AdamW | warmup_cosine | 66 | 93 | 89 |
| 02 | densenet | no | 128x128x128 | yes | CE | 92/150 | 0.05 | AdamW | warmup_cosine | 99 | 75 | 79 |
| 03 | densenet | yes | 128x128x128 | no | CE | 150/150 | 0.05 | AdamW | warmup_cosine | 60 | 89 | 86 |
| 04 | densenet | yes | 64x64x64 | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 45 | 77 | 80 |
| 05 | densenet | yes | 64x64x64 | no | CE | 110/150 | 0.005 | AdamW | warmup_cosine | 45 | 81 | 82 |
| 06 | densenet | yes | 64x64x64 | no | CE | 114/150 | 0.0005 | AdamW | warmup_cosine | 48 | 56 | 46 |
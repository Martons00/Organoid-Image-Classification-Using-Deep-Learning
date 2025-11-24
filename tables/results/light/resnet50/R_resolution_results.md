## Experiment Results outputs/OrganoidsINRIA_reduced/resnet50

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | resnet50 | yes | 128x128x128 | no | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 80 | 60 | 65 |
| 02 | resnet50 | yes | 128x128x128 | yes | CE | 82/150 | 0.0005 | AdamW | warmup_cosine | 78 | 78 | 74 |
| 03 | resnet50 | yes | 128x128x128 | no | CE | 90/150 | 0.001 | AdamW | warmup_cosine | 82 | 66 | 59 |
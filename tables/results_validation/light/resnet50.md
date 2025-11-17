## Experiment Results outputs/OrganoidsINRIA/resnet50

| Run | Model | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | resnet50 | no | FocalLoss | 100 | 0.0006 | AdamW | warmup_cosine | yes | percentage | 1 | 39 |
| 02 | resnet50 | yes | CE | 100 | 0.0003 | AdamW | cosine_anneal | yes | percentage | 92 | 46 |
| 03 | resnet50 | yes | FocalLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | percentage | 36 | 46 |
| 04 | resnet50 | yes | FocalLoss | 150 | 0.0003 | AdamW | warmup_cosine | yes | balanced | 39 | 51 |
| 05_fc | resnet50 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 38 | 44 |
| 06_layer_4_2_fc | resnet50 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 82 | 46 |
| 07_layer4_fc | resnet50 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 96 | 55 |
| 08 | resnet50 | yes | FocalLoss | 4 | 2e-05 | AdamW | warmup_cosine | no | percentage | * | * |
| 09 | resnet50 | yes | CE | 100 | 0.0001 | AdamW | warmup_cosine | no | stratified | * | * |
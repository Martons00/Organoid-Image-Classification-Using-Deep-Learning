## Experiment Results outputs/OrganoidsINRIA/resnet50

| Run | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | FocalLoss | 100 | 0.0006 | AdamW | warmup_cosine | yes | percentage | 1 | 39 |
| 02 | CE | 100 | 0.0003 | AdamW | cosine_anneal | yes | percentage | 92 | 46 |
| 03 | FocalLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | percentage | 36 | 46 |
| 04 | FocalLoss | 150 | 0.0003 | AdamW | warmup_cosine | yes | balanced | 39 | 51 |
| 05_fc | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 38 | 44 |
| 06_layer_4_2_fc | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 82 | 46 |
| 07_layer4_fc | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 96 | 55 |
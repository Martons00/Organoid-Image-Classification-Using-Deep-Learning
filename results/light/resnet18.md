## Experiment Results outputs/OrganoidsINRIA/resnet18

| Run | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | FocalLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | percentage | 49 | 41 |
| 02 | FocalLoss | 150 | 0.0004 | AdamW | warmup_cosine_restarts | yes | percentage | 43 | 48 |
| 03 | FocalLoss | 200 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 59 | 53 |
| 04 | FocalLoss | 200 | 0.0005 | AdamW | warmup_cosine_restarts | yes | percentage | 49 | 51 |
| 05 | FocalLoss | 200 | 0.005 | AdamW | warmup_cosine | yes | percentage | 61 | 51 |
| 06 | FocalLoss | 200 | 0.005 | AdamW | warmup_cosine_restarts | yes | percentage | 61 | 56 |
| 07 | FocalLoss | 200 | 0.05 | AdamW | warmup_cosine_restarts | yes | percentage | 73 | 46 |
| 08 | FocalLoss | 200 | 0.01 | AdamW | warmup_cosine | yes | percentage | 71 | 58 |
| 09 | FocalLoss | 200 | 0.05 | AdamW | warmup_cosine | yes | percentage | 80 | 70 |
| 10 | CE | 200 | 0.1 | AdamW | warmup_cosine | yes | percentage | 80 | 65 |
## Experiment Results outputs/OrganoidsINRIA/resnet18

| Run | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | yes | FocalLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | percentage | 49 | 41 |
| 02 | yes | FocalLoss | 150 | 0.0004 | AdamW | warmup_cosine_restarts | yes | percentage | 43 | 48 |
| 03 | yes | FocalLoss | 200 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 59 | 53 |
| 04 | yes | FocalLoss | 200 | 0.0005 | AdamW | warmup_cosine_restarts | yes | percentage | 49 | 51 |
| 05 | yes | FocalLoss | 200 | 0.005 | AdamW | warmup_cosine | yes | percentage | 61 | 51 |
| 06 | yes | FocalLoss | 200 | 0.005 | AdamW | warmup_cosine_restarts | yes | percentage | 61 | 56 |
| 07 | yes | FocalLoss | 200 | 0.05 | AdamW | warmup_cosine_restarts | yes | percentage | 73 | 46 |
| 08 | yes | FocalLoss | 200 | 0.01 | AdamW | warmup_cosine | yes | percentage | 71 | 58 |
| 09 | yes | FocalLoss | 200 | 0.05 | AdamW | warmup_cosine | yes | percentage | 80 | 70 |
| 10 | yes | CE | 200 | 0.1 | AdamW | warmup_cosine | yes | percentage | 80 | 65 |
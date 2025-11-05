## Experiment Results outputs/OrganoidsINRIA/swinunetr/encoder10+fc

| Run | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | yes | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 51 | 74 |
| 02 | yes | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 50 | 74 |
| 03 | yes | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 49 | 74 |
| 04 | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 60 | 59 |
| 05_brat_model | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 77 | 59 |
| 06 | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 50 | 63 |
| 07 | yes | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 55 |
| 08 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 59 |
| 09 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 62 |
| 10 | yes | FocalLoss | 100 | 0.0003 | AdamW | warmup_cosine | no | percentage | 68 | 76 |
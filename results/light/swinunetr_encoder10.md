## Experiment Results outputs/OrganoidsINRIA/swinunetr/encoder10+fc

| Run | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 51 | 74 |
| 02 | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 50 | 74 |
| 03 | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 49 | 74 |
| 04 | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 60 | 59 |
| 05_brat_model | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 77 | 59 |
| 06 | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 50 | 63 |
| 07 | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 55 |
| 08 | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 59 |
| 09 | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 62 |
| 10 | FocalLoss | 100 | 0.0003 | AdamW | warmup_cosine | no | percentage | 68 | 76 |
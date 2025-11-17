## Experiment Results outputs/OrganoidsINRIA/swinunetr/encoder10+fc

| Run | Model | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinunetr(e) | yes | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 51 | 74 |
| 02 | swinunetr(e) | yes | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 50 | 74 |
| 03 | swinunetr(e) | yes | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 49 | 74 |
| 04 | swinunetr(e) | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 60 | 59 |
| 05_brat_model | swinunetr(e) | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 77 | 59 |
| 06 | swinunetr(e) | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 50 | 63 |
| 07 | swinunetr(e) | yes | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 55 |
| 08 | swinunetr(e) | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 59 |
| 09 | swinunetr(e) | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 62 |
| 10 | swinunetr(e) | yes | FocalLoss | 100 | 0.0003 | AdamW | warmup_cosine | no | percentage | 68 | 76 |
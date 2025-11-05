## Experiment Results outputs/OrganoidsINRIA/swinunetr/layer4+encoder10+fc

| Run | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | CombinedLoss | 100 | 0.001 | AdamW | cosine_anneal | yes | balanced | 47 | 70 |
| 02 | CE | 100 | 0.001 | AdamW | cosine_anneal | yes | percentage | 65 | 73 |
| 03 | CE | 100 | 0.001 | AdamW | cosine_anneal | yes | percentage | 81 | 79 |
| 04 | CE | 100 | 0.0003 | AdamW | cosine_anneal | yes | percentage | 86 | 76 |
| 05 | CE | 100 | 0.0003 | AdamW | warmup_cosine | yes | percentage | 88 | 79 |
| 06 | CE | 100 | 0.0003 | AdamW |  | yes | percentage | 84 | 85 |
| 07 | FocalLoss | 100 | 0.0003 | AdamW | warmup_cosine | yes | percentage | 88 | 79 |
| 08 | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | no | percentage | 92 | 81 |
| 09 | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | no | percentage | 93 | 84 |
| 10 | CE | 50 | 0.0003 | AdamW |  | no | stratified | 75 | 80 |
| 11 | CE | 50 | 0.0003 | AdamW |  | yes | stratified | 80 | 80 |
## Experiment Results outputs/OrganoidsINRIA/swinunetr/layer4+encoder10+fc

| Run | Model | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | swinunetr | yes | CombinedLoss | 100 | 0.001 | AdamW | cosine_anneal | yes | balanced | 47 | 70 |
| 02 | swinunetr | yes | CE | 100 | 0.001 | AdamW | cosine_anneal | yes | percentage | 65 | 73 |
| 03 | swinunetr | yes | CE | 100 | 0.001 | AdamW | cosine_anneal | yes | percentage | 81 | 79 |
| 04 | swinunetr | yes | CE | 100 | 0.0003 | AdamW | cosine_anneal | yes | percentage | 86 | 76 |
| 05 | swinunetr | yes | CE | 100 | 0.0003 | AdamW | warmup_cosine | yes | percentage | 88 | 79 |
| 06 | swinunetr | yes | CE | 100 | 0.0003 | AdamW |  | yes | percentage | 84 | 85 |
| 07 | swinunetr | yes | FocalLoss | 100 | 0.0003 | AdamW | warmup_cosine | yes | percentage | 88 | 79 |
| 08 | swinunetr | yes | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | no | percentage | 92 | 81 |
| 09 | swinunetr | yes | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | no | percentage | 93 | 84 |
| 10 | swinunetr | yes | CE | 50 | 0.0003 | AdamW |  | no | stratified | 75 | 80 |
| 11 | swinunetr | yes | CE | 50 | 0.0003 | AdamW |  | yes | stratified | 80 | 80 |
| 12 | swinunetr | yes | CE | 60 | 0.0003 | AdamW | warmup_cosine | no | stratified | 82 | 86 |
| 13 | swinunetr | yes | CE | 60 | 0.0003 | AdamW | warmup_cosine | no | stratified | * | * |
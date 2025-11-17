## Experiment Results outputs/OrganoidsINRIA/swinunetr/layer4+encoder10+fc

| Run | Model | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | swinunetr | yes | CE | 60 | 0.0003 | AdamW | warmup_cosine | no | stratified | 82 | 86 |
| 06 | swinunetr | yes | CE | 100 | 0.0003 | AdamW |  | yes | percentage | 84 | 85 |
| 09 | swinunetr | yes | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | no | percentage | 93 | 84 |
| 08 | swinunetr | yes | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | no | percentage | 92 | 81 |
| 10 | swinunetr | yes | CE | 50 | 0.0003 | AdamW |  | no | stratified | 75 | 80 |
| 11 | swinunetr | yes | CE | 50 | 0.0003 | AdamW |  | yes | stratified | 80 | 80 |
| 03 | swinunetr | yes | CE | 100 | 0.001 | AdamW | cosine_anneal | yes | percentage | 81 | 79 |
| 05 | swinunetr | yes | CE | 100 | 0.0003 | AdamW | warmup_cosine | yes | percentage | 88 | 79 |
| 07 | swinunetr | yes | FocalLoss | 100 | 0.0003 | AdamW | warmup_cosine | yes | percentage | 88 | 79 |
| 04 | swinunetr | yes | CE | 100 | 0.0003 | AdamW | cosine_anneal | yes | percentage | 86 | 76 |
| 10 | swinunetr(e) | yes | FocalLoss | 100 | 0.0003 | AdamW | warmup_cosine | no | percentage | 68 | 76 |
| 01 | swinunetr(e) | yes | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 51 | 74 |
| 02 | swinunetr(e) | yes | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 50 | 74 |
| 03 | swinunetr(e) | yes | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 49 | 74 |
| 02 | swinunetr | yes | CE | 100 | 0.001 | AdamW | cosine_anneal | yes | percentage | 65 | 73 |
| 01 | swinunetr | yes | CombinedLoss | 100 | 0.001 | AdamW | cosine_anneal | yes | balanced | 47 | 70 |
| 09 | resnet18 | yes | FocalLoss | 200 | 0.05 | AdamW | warmup_cosine | yes | percentage | 80 | 70 |
| 10 | resnet18 | yes | CE | 200 | 0.1 | AdamW | warmup_cosine | yes | percentage | 80 | 65 |
| 06 | swinunetr(e) | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 50 | 63 |
| 09 | swinunetr(e) | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 62 |
| 04 | swinunetr(e) | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 60 | 59 |
| 05_brat_model | swinunetr(e) | no | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 77 | 59 |
| 08 | swinunetr(e) | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 59 |
| 08 | resnet18 | yes | FocalLoss | 200 | 0.01 | AdamW | warmup_cosine | yes | percentage | 71 | 58 |
| 06 | resnet18 | yes | FocalLoss | 200 | 0.005 | AdamW | warmup_cosine_restarts | yes | percentage | 61 | 56 |
| 07 | swinunetr(e) | yes | FocalLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | balanced | - | 55 |
| 07_layer4_fc | resnet50 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 96 | 55 |
| 03 | resnet18 | yes | FocalLoss | 200 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 59 | 53 |
| 04 | resnet18 | yes | FocalLoss | 200 | 0.0005 | AdamW | warmup_cosine_restarts | yes | percentage | 49 | 51 |
| 05 | resnet18 | yes | FocalLoss | 200 | 0.005 | AdamW | warmup_cosine | yes | percentage | 61 | 51 |
| 04 | resnet50 | yes | FocalLoss | 150 | 0.0003 | AdamW | warmup_cosine | yes | balanced | 39 | 51 |
| 02 | resnet18 | yes | FocalLoss | 150 | 0.0004 | AdamW | warmup_cosine_restarts | yes | percentage | 43 | 48 |
| 11 | resnet18 | yes | CE | 100 | 0.1 | AdamW | warmup_cosine | no | stratified | 71 | 48 |
| 07 | resnet18 | yes | FocalLoss | 200 | 0.05 | AdamW | warmup_cosine_restarts | yes | percentage | 73 | 46 |
| 02 | resnet50 | yes | CE | 100 | 0.0003 | AdamW | cosine_anneal | yes | percentage | 92 | 46 |
| 03 | resnet50 | yes | FocalLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | percentage | 36 | 46 |
| 06_layer_4_2_fc | resnet50 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 82 | 46 |
| 05_fc | resnet50 | yes | CombinedLoss | 150 | 0.0005 | AdamW | warmup_cosine | yes | percentage | 38 | 44 |
| 01 | resnet18 | yes | FocalLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | percentage | 49 | 41 |
| 01 | resnet50 | no | FocalLoss | 100 | 0.0006 | AdamW | warmup_cosine | yes | percentage | 1 | 39 |
| 13 | swinunetr | yes | CE | 60 | 0.0003 | AdamW | warmup_cosine | no | stratified | * | 0 |
| 08 | resnet50 | yes | FocalLoss | 4 | 2e-05 | AdamW | warmup_cosine | no | percentage | * | 0 |
| 09 | resnet50 | yes | CE | 100 | 0.0001 | AdamW | warmup_cosine | no | stratified | * | 0 |

## Experiment Results outputs/OrganoidsINRIA_reduced/resnet18

| Run | Model | Aug | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | TrainAcc | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | resnet18 | yes | CE | 100 | 0.1 | AdamW | warmup_cosine | no | stratified | 66 | 52 |
| 02 | resnet18 | yes | CE | 100 | 0.05 | AdamW | warmup_cosine | no | stratified | 96 | 52 |
| 03 | resnet18 | yes | CE | 100 | 0.01 | AdamW | warmup_cosine | no | stratified | 66 | 38 |
| 01 | resnet50 | yes | CE | 100 | 0.0001 | AdamW | warmup_cosine | no | stratified | 49 | 61 |
| 02 | resnet50 | yes | CE | 100 | 0.0001 | AdamW | warmup_cosine | no | stratified | 76 | 56 |
| 01 | swinunetr | yes | CE | 60 | 0.0003 | AdamW | warmup_cosine | no | stratified | 90 | 88 |
| 02 | swinunetr | yes | CE | 60 | 0.0006 | AdamW | warmup_cosine | no | stratified | 89 | 87 |
| 03 | swinunetr | yes | CE | 60 | 0.0006 | AdamW | warmup_cosine | no | stratified | 86 | 88 |
| 04 | swinunetr | yes | CE | 60 | 0.001 | AdamW | warmup_cosine | no | stratified | 87 | 84 |
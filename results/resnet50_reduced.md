## Experiment Results outputs/OrganoidsINRIA_reduced/resnet50
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainLoss | TrainAcc | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-11-07 | 01 | resnet50 | yes | no | 128x128x128 | CE | none | yes | no | 4 | 100 | 4 | 0.0001 | AdamW | 0.99 | warmup_cosine | no | no | stratified | 1 | 49 | 61 | 59 | 57 | 61 | 65 | (SOLO FC) Come resnet18 si nota proprio il fatto che con questa risoluzione di samples, queste CNN riescano a comprendere meglio il contesto e a generalizzare di più. La loss è rumorosa, forse meglio diminuire il LR. |
| 2025-11-10 | 02 | resnet50 | yes | no | 128x128x128 | CE | none | yes | no | 4 | 100 | 4 | 0.0001 | AdamW | 0.99 | warmup_cosine | no | no | stratified | 0,55 | 76 | 56 | 55 | 58 | 56 | 63 | (LAYER4) Non si converte in performance la conoscenza in validation |



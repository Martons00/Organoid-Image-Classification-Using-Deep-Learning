## Experiment Results outputs/OrganoidsINRIA_reduced/swinvit
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainLoss | TrainAcc | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | TestAcc | W_TestF1 | W_TestPrecision | W_TestRecall | W_TestSpecificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-11-21 | 01 | swinvit | yes | no | 128x128x128 | CE | none | yes | no | 6 | */150 | 5 | 0.001 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | * | * | * | * | * | * | * | * | * | * | * | * | * |
| 2025-11-24 | 02 | swinvit | yes | no | 128x128x128 | CE | none | yes | no | 6 | */150 | 5 | 0.006 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | * | * | * | * | * | * | * | * | * | * | * | * | * |
| 2025-11-25 | 2025-11-25-09-38 | swinvit | yes | no | 128x128x128 | CE | none | yes | no | 6 | */300 | 5 | 0.01 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | * | * | * | * | * | * | * | * | * | * | * | * | * |



## Experiment Results outputs/OrganoidsINRIA/resnet50
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainLoss | TrainAcc | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | TestAcc | W_TestF1 | W_TestPrecision | W_TestRecall | W_TestSpecificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-11-16 | 01 | resnet50 | yes | no | 128x128x128 | CE | none | yes | no | 4 | 84/150 | 4 | 0.0005 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 0,39 | 50 | 50 | 34 | 69 | 50 | 52 | 52 | 41 | 70 | 53 | 55 | Qui la loss scende bene, forse si può usare un LR più alto. Il problema però è sempre il medesimo Non convertono la conoscenza in training con conoscenza su samples non visti.  |



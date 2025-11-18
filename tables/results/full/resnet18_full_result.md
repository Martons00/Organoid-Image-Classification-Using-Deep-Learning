## Experiment Results outputs/OrganoidsINRIA/resnet18
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainLoss | TrainAcc | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | TestAcc | W_TestF1 | W_TestPrecision | W_TestRecall | W_TestSpecificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-11-15 | 01 | resnet18 | yes | no | 128x128x128 | CE | none | yes | no | 6 | 150*/150 | 5 | 0.05 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 0,02 | 99 | 63 | 61 | 63 | 63 | 68 | 72 | 69 | 74 | 72 | 76 | Stesso solito atteggiamento delle CNN, non capitscono molto qunado la risoluzione è grande. Se si diminuisce la risoluzione invece va meglio. C’è però da dire che la loss è rumorosa quindi si potrebbe fare un altro tentativo magari. L’overfitting è molto alto però.  |



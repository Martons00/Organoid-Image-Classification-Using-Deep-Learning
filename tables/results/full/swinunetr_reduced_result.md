## Experiment Results outputs/OrganoidsINRIA_reduced/swinunetr
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainLoss | TrainAcc | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | TestAcc | W_TestF1 | W_TestPrecision | W_TestRecall | W_TestSpecificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-11-15 | 01 | swinunetr | yes | no | 128x128x128 | CE | none | yes | no | 6 | 142/150 | 5 | 0.0006 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 0,18 | 91 | 86 | 86 | 86 | 86 | 89 | 90 | 90 | 90 | 90 | 93 | Andamento buono, le curve sono buone, forse un LR leggermente più alto si può raggiungere una performance migliore.  |
| 2025-11-15 | 02 | swinunetr | yes | no | 128x128x128 | CE | none | yes | no | 6 | 142*/150 | 5 | 0.0006 | AdamW | 0.99 | warmup_cosine | yes | yes | stratified | 0,32 | 86 | 87 | 87 | 87 | 87 | 90 | 86 | 86 | 88 | 86 | 90 | Le performance sono buone, sembra essere quasi arrivato al massimo delle performance raggiungibili.  |



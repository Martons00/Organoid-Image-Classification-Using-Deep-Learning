## Complete Results

| Rank | Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainLoss | TrainAcc | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | TestAcc | W_TestF1 | W_TestPrecision | W_TestRecall | W_TestSpecificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2025-12-12 | 01_L | resnet18 | yes | no | 128x128x128 | CE | none | yes | no | 6 | */150 | 5 | 0.05 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 0.0524 | 97 | 93 | 93 | 94 | 93 | 97 | 94 | 94 | 94 | 94 | 96 | * |
| 2 | 2025-12-11 | 01_L | densenet | yes | no | 128x128x128 | CE | none | yes | no | 2 | */150 | 5 | 0.01 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 0.4342 | 64 | 91 | 91 | 91 | 91 | 94 | 89 | 89 | 90 | 89 | 94 | * |
| 3 | 2025-12-12 | 01_L | swinunetr | yes | no | 128x128x128 | CE | none | yes | no | 6 | */150 | 5 | 0.001 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 0.2844 | 87 | 83 | 84 | 84 | 83 | 89 | 88 | 88 | 89 | 88 | 91 | * |
| 4 | 2025-12-12 | 01_L | swinunetr+noah | yes | no | 128x128x128 | CE | none | yes | no | 6 | */150 | 5 | 0.0006 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 0.3586 | 82 | 84 | 84 | 85 | 84 | 86 | 79 | 79 | 80 | 79 | 85 | * |
| 5 | 2025-12-12 | 01_L | swinvit | yes | no | 128x128x128 | CE | none | yes | no | 6 | */300 | 5 | 0.05 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 1.5085 | 39 | 49 | 32 | 24 | 49 | 51 | 49 | 32 | 24 | 49 | 51 | * |
| 6 | 2025-12-12 | 02_L | swinvit | yes | no | 128x128x128 | CE | none | yes | no | 6 | */300 | 5 | 0.5 | AdamW | 0.99 | warmup_cosine | yes | no | stratified | 4.9455 | 38 | 49 | 32 | 24 | 49 | 51 | 49 | 32 | 24 | 49 | 51 | * |

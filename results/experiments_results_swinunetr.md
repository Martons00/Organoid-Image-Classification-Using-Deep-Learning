## Experiment Results outputs/OrganoidsINRIA/swinunetr/layer4+encoder10+fc
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainAcc | TrainLoss | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-27 | 01 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 6 | 100 | 4 | 0.001 | AdamW | 0.99 | cosine_anneal | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-27 | 02 | swinunetr | yes | yes | 128x128x128 | CE | none | yes | no | 6 | 100 | 4 | 0.001 | AdamW | 0.99 | cosine_anneal | no | no | percentage | - | - | - | - | - | - | - | ... |
| 2025-10-28 | 03 | swinunetr | yes | yes | 128x128x128 | CE | none | yes | no | 4 | 100 | 4 | 0.001 | AdamW | 0.99 | cosine_anneal | no | yes | percentage | - | - | - | - | - | - | - | ... |
| 2025-10-28 | 04 | swinunetr | yes | yes | 128x128x128 | CE | none | yes | no | 6 | 100 | 8 | 0.0003 | AdamW | 0.99 | cosine_anneal | no | no | percentage | - | - | - | - | - | - | - | ... |
| 2025-10-28 | 05 | swinunetr | yes | yes | 128x128x128 | CE | none | yes | no | 6 | 100 | 8 | 0.0003 | AdamW | 0.99 | warmup_cosine | no | no | percentage | - | - | - | - | - | - | - | ... |
| 2025-10-28 | 06 | swinunetr | yes | yes | 128x128x128 | CE | none | yes | no | 6 | 100 | 8 | 0.0003 | AdamW | 0.99 |  | no | no | percentage | - | - | - | - | - | - | - | ... |
| 2025-10-28 | 07 | swinunetr | yes | yes | 128x128x128 | FocalLoss | none | yes | no | 6 | 100 | 8 | 0.0003 | AdamW | 0.99 | warmup_cosine | no | no | percentage | - | - | - | - | - | - | - | ... |


## Experiment Results outputs/OrganoidsINRIA/swinunetr/encoder10+fc
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | TrainAcc | TrainLoss | ValAcc | W_ValF1 | W_ValPrecision | W_ValRecall | W_Specificity | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-24 | 01 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 4 | 0.001 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-24 | 02 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 4 | 0.001 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-24 | 03 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 4 | 0.0005 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-24 | 04 | swinunetr | no | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 4 | 0.0005 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-25 | 05_brat_model | swinunetr | no | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 4 | 0.0005 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-24 | 06 | swinunetr | no | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 4 | 0.0005 | AdamW | 0.99 | warmup_cosine | no | yes | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-22 | 07 | swinunetr | yes | yes | 128x128x128 | FocalLoss | none | yes | no | 8 | 150 | 7 | 0.0005 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-22 | 08 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 150 | 7 | 0.0005 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-22 | 09 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 150 | 7 | 0.0005 | AdamW | 0.99 | warmup_cosine | no | no | balanced | - | - | - | - | - | - | - | ... |
| 2025-10-29 | 10 | swinunetr | yes | no | 128x128x128 | FocalLoss | none | yes | no | 6 | 100 | 8 | 0.0003 | AdamW | 0.99 | warmup_cosine | no | no | percentage | - | - | - | - | - | - | - | ... |



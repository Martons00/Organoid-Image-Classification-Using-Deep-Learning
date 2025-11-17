## Experiment Results outputs/OrganoidsINRIA_old/swinunetr

| Run | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | - | 50 | 0.001 | AdamW | warmup_cosine | yes | balanced | 70 |
| 02 | CombinedLoss | 100 | 0.001 | AdamW | cosine_anneal | yes | balanced | 77 |
| 03 | CombinedLoss | 60 | 0.001 | AdamW | warmup_cosine | yes | balanced | 77 |
| 04 | CombinedLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | balanced | 74 |
| 05 | CombinedLoss | 150 | 0.0001 | AdamW | warmup_cosine | yes | balanced | 74 |
| 06 | CE | 60 | 5e-05 | AdamW | warmup_cosine | yes | balanced | 44 |
| 07 | FocalLoss | 100 | 0.0005 | AdamW | cosine_anneal | yes | balanced | 55 |
| 08 | CombinedLoss | 100 | 0.0005 | AdamW | cosine_anneal | yes | balanced | 59 |
| 09 | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 74 |
| 10 | CombinedLoss | 100 | 0.0005 | AdamW | cosine_anneal | yes | balanced | 77 |
| 11 | FocalLoss | 100 | 0.0005 | AdamW | cosine_anneal | yes | balanced | 77 |
| 12 | CombinedLoss | 100 | 0.0005 | AdamW | cosine_anneal | yes | balanced | 81 |

## Experiment Results outputs/OrganoidsINRIA_old/swinunetr MLDECODER

| Run | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | FocalLoss | 100 | 0.005 | AdamW | cosine_anneal | yes | balanced | 33 |
| 02 | FocalLoss | 300 | 0.005 | AdamW | cosine_anneal | yes | balanced | 33 |
| 03 | FocalLoss | 300 | 0.0005 | AdamW | cosine_anneal | yes | balanced | 59 |

## Experiment Results outputs/OrganoidsINRIA_old/swinunetr NOAH

| Run | Loss | MaxEpochs | LR | Optim | LRschedule | ExactClass | SplitMethod | ValAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | CombinedLoss | 150 | 0.001 | AdamW | warmup_cosine | yes | balanced | 55 |
| 02 | CombinedLoss | 100 | 0.001 | AdamW | warmup_cosine | yes | balanced | 59 |
| 03 | CombinedLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 62 |
| 04 | FocalLoss | 100 | 0.0005 | AdamW | warmup_cosine | yes | balanced | 59 |
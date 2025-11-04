Alcuni esperimenti sono vecchi, quindi ad alcuni mancano i plot, le metriche, il salvataggio dei modelli, ad alcuni c'era un problema nel caricamento della backbone pretrainato
**Nota:** Risoluzione dei problemi di caricamento della backbone pretrainato 2025-10-14

## Experiment Results outputs/OrganoidsINRIA_old/swinunetr
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | ValAcc | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-10 | 01 | swinunetr | yes | yes | 128x128x128 | - | none | yes | no | 1 | 50 | 5 | 0.001 | AdamW | 0.99 | warmup_cosine | no | no | balanced | 70 | ... |
| 2025-10-16 | 02 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 3 | 0.001 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 77 | killed 70 epoch |
| 2025-10-14 | 03 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 60 | 3 | 0.001 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 77 | ... |
| 2025-10-21 | 04 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 150 | 3 | 0.0001 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 74 | ... |
| 2025-10-21 | 05 | swinunetr | yes | yes | 256x256x128 | CombinedLoss | none | yes | no | 8 | 150 | 3 | 0.0001 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 74 | killed 116 epoch |
| 2025-10-13 | 06 | swinunetr | yes | yes | 128x128x128 | - | none | yes | no | 1 | 60 | 3 | 5e-05 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 44 | killed 9 epoch |
| 2025-10-21 | 07 | swinunetr | yes | yes | 128x128x128 | FocalLoss | margin (w=0.5) | yes | no | 6 | 100 | 10 | 0.0005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 55 | la sim loss non funzionaa bene con questa w |
| 2025-10-21 | 08 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | margin (w=0.5) | yes | no | 6 | 100 | 10 | 0.0005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 59 | idem |
| 2025-10-15 | 09 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 3 | 0.0005 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 74 | early stop 83 epoch |
| 2025-10-16 | 10 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 10 | 0.0005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 77 | ... |
| 2025-10-20 | 11 | swinunetr | yes | yes | 128x128x128 | FocalLoss | none | yes | no | 8 | 100 | 10 | 0.0005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 77 | ... |
| 2025-10-15 | 12 | swinunetr | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 3 | 0.0005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 81 | early sop 79 |


## Experiment Results outputs/OrganoidsINRIA_old/swinunetr+ml_decoder
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | ValAcc | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-20 | 01 | swinunetr+ml_decoder | yes | yes | 128x128x128 | FocalLoss | none | yes | no | 8 | 100 | 10 | 0.005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 33 | ... |
| 2025-10-21 | 02 | swinunetr+ml_decoder | yes | yes | 128x128x128 | FocalLoss | none | yes | no | 8 | 300 | 10 | 0.005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 33 | ... |
| 2025-10-21 | 03 | swinunetr+ml_decoder | yes | yes | 128x128x128 | FocalLoss | none | yes | no | 8 | 300 | 10 | 0.0005 | AdamW | 0.99 | cosine_anneal | yes | no | balanced | 59 | killed 112 epoch |


## Experiment Results outputs/OrganoidsINRIA_old/swinunetr+noah
| Date | Run | Model | Aug | ExactClass | ROI | Loss | SimLoss | Checkpoint | Encoder10 | Batch | MaxEpochs | Warmup | LR | Optim | Momentum | LRschedule | EarlyStop | PatchMerging | SplitMethod | ValAcc | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-21 | 01 | swinunetr+noah | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 150 | 3 | 0.001 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 55 | ... |
| 2025-10-20 | 02 | swinunetr+noah | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 3 | 0.001 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 59 | ... |
| 2025-10-15 | 03 | swinunetr+noah | yes | yes | 128x128x128 | CombinedLoss | none | yes | no | 8 | 100 | 3 | 0.0005 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 62 | ... |
| 2025-10-20 | 04 | swinunetr+noah | yes | yes | 128x128x128 | FocalLoss | none | yes | no | 8 | 100 | 3 | 0.0005 | AdamW | 0.99 | warmup_cosine | yes | no | balanced | 59 | ... |


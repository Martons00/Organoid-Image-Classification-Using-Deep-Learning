## Full Results

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | densenet | no | 128x128x128 | no | CE | 88/150 | 0.05 | AdamW | warmup_cosine | 47 | 50 | 53 |
| 02 | densenet | yes | 128x128x128 | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 50 | 50 | 53 
| 01 | resnet18 | yes | 128x128x128 | no | CE | 150*/150 | 0.05 | AdamW | warmup_cosine | 99 | 63 | 72 
| 01 | resnet50 | yes | 128x128x128 | no | CE | 84/150 | 0.0005 | AdamW | warmup_cosine | 50 | 50 | 52 
| 01 | swinunetr | yes | 128x128x128 | no | CE | 118/150 | 0.0006 | AdamW | warmup_cosine | 85 | 82 | 90 
| 01 | swinunetr+noah | yes | 128x128x128 | no | CE | 106/150 | 0.0006 | AdamW | warmup_cosine | 75 | 78 | 85 
| 01 | swinvit | yes | 128x128x128 | no | CE | 164/300 | 0.006 | AdamW | warmup_cosine | 82 | 80 | 85 |
| 02 | swinvit | yes | 128x128x128 | no | CE | 196/300 | 0.04 | AdamW | warmup_cosine | 73 | 74 | 79 

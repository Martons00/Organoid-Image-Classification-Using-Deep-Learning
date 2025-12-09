## Full Results

| Run | Model | Aug | ROI | PatchMerging | Loss | MaxEpochs | LR | Optim | LRschedule | TrainAcc | ValAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_F | densenet | no | 128x128x128 | no | CE | 88/150 | 0.05 | AdamW | warmup_cosine | 47 | 50 | 53 |
| 02_F  | densenet | yes | 128x128x128 | no | CE | 86/150 | 0.05 | AdamW | warmup_cosine | 50 | 50 | 53 
| 01_F  | resnet18 | yes | 128x128x128 | no | CE | 150*/150 | 0.05 | AdamW | warmup_cosine | 99 | 63 | 72 
| 01_F  | resnet50 | yes | 128x128x128 | no | CE | 84/150 | 0.0005 | AdamW | warmup_cosine | 50 | 50 | 52 
| 01_F  | swinunetr | yes | 128x128x128 | no | CE | 118/150 | 0.0006 | AdamW | warmup_cosine | 85 | 82 | 90 
| 01_F  | swinunetr+noah | yes | 128x128x128 | no | CE | 106/150 | 0.0006 | AdamW | warmup_cosine | 75 | 78 | 85 
| 01_F  | swinvit | yes | 128x128x128 | no | CE | 164/300 | 0.006 | AdamW | warmup_cosine | 82 | 80 | 85 |
| 02_F  | swinvit | yes | 128x128x128 | no | CE | 196/300 | 0.04 | AdamW | warmup_cosine | 73 | 74 | 79 |

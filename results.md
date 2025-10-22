
| Gruppo                | LOSS                                         | LR    | Epoche     | Val | Scheduler |
|---                    |---                                           |---    |---         |---  |---        |
| swinunetr             | (Focal + Diversity Loss)                     | 1E-3  | 60         | 70  | cosine_anneal |
| swinunetr             | (Focal + Diversity Loss)                     | 1E-3  | 60         | 77  | warmup |
| swinunetr             | (Focal + Diversity Loss)                     | 1E-3  | 100        | 74  | warmup |
| swinunetr             | (Focal + Diversity Loss)                     | 1E-4  | 60         | 74  | warmup |
| swinunetr (pt)        | (Focal + Diversity Loss)                     | 1E-4  | 79         | 74  | warmup |
| swinunetr (256x256)   | (Focal + Diversity Loss)                     | 1E-4  | 115        | 67  | warmup |
| swinunetr             | (Focal + Diversity Loss)                     | 1E-5  | 60         | 66  | warmup |
| swinunetr (sim_m)     | (Focal + Diversity Loss + Similarity Margin) | 5E-4  | 100        | 55  | sim_marg_focal |
| swinunetr (sim_c)     | (Focal + Diversity Loss + Similarity Contr.) | 5E-4  | 100        | 56  | sim_marg_combined |
| swinunetr             | (Focal + Diversity Loss)                     | 5E-4  | 100        | 74  | warmup |
| swinunetr             | (Focal + Diversity Loss)                     | 5E-4  | 100        | 77  | cosine_anneal |
| swinunetr (pt)        | focal loss                                   | 5E-4  | 100        | 77  | warmup |
| swinunetr             | (Focal + Diversity Loss)                     | 5E-4  | 100        | 81  | cosine_anneal |

| Gruppo                | LOSS                                         | LR    | Epoche     | Val | Scheduler |
|---                    |---                                           |---    |---         |---  |---        |
| swinunetr+ml_decoder  | focal loss                                   | 5E-3  | 43         | 33  | cosine_anneal |
| swinunetr+ml_decoder  | focal loss                                   | 5E-3  | 111        | 33  | cosine_anneal |
| swinunetr+ml_decoder  | focal loss                                   | 5E-4  | 111        | 59  | cosine_anneal |

| Gruppo                | LOSS                                         | LR    | Epoche     | Val | Scheduler |
|---                    |---                                           |---    |---         |---  |---        |
| swinunetr+noah (pt)   | (Focal + Diversity Loss)                     | 1E-3  | 110-150    | 55  | warmup |
| swinunetr+noah (pt)   | (Focal + Diversity Loss)                     | 1E-4  | 110-200    | 44  | warmup |
| swinunetr+noah        | (Focal + Diversity Loss)                     | 1E-3  | 51-99      | 59  | warmup |
| swinunetr+noah        | (Focal + Diversity Loss)                     | 1E-6  | 93-99      | 62  | warmup |
| swinunetr+noah (pt)   | (Focal + Diversity Loss)                     | 5E-4  | 90-300     | 44  | warmup |
| swinunetr+noah (pt)   | focal loss                                   | 5E-4  | 51-99      | 59  | warmup |

### Cluster execution
- Testing with cluster scripts using best effort and priority 1
- Distributed testing across multiple GPUs
- Telegram bot for cluster management

### Models and weights
- Pre-trained model: SwinViT
- Pre-trained encoder10
- Model selection: 'swinunetr', 'swinunetr+ml_decoder', or 'swinunetr+noah'

### Data and splits
- Dataset split method: 'random', 'stratified', or 'balanced'
- Training set split method: 'random' or 'balanced'

### Configuration and control
- Automated configuration via YAML file
- Checkpoint
- Early stopping
- Debug mode
- Telegram bot for monitoring

### Optimization and losses
- Scheduler: 'cosine_anneal' or 'warmup_cosine'
- Optimizer: 'adam', 'adamw', or 'sgd'
- Loss: 'CE', 'FocalLoss', 'LabelSmoothingLoss', 'DiversityLoss', 'CombinedLoss', or 'CenterLoss'
- Similarity loss: 'contrastive', 'margin', or '' (none)
import torch
import torch.nn as nn

class ResNet_3D(nn.Module):
    def __init__(self, original_model: nn.Module, num_classes: int = 3, num_features: int = 2048):
        super().__init__()
        # Copia backbone 3D
        self.conv1   = original_model.conv1
        self.bn1     = original_model.bn1
        self.relu    = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1  = original_model.layer1
        self.layer2  = original_model.layer2
        self.layer3  = original_model.layer3
        self.layer4  = original_model.layer4

        # Mantieni la testa di segmentazione solo se ti serve altrove (non usata per la classificazione)
        self.has_seg_head = hasattr(original_model, "conv_seg")
        if self.has_seg_head:
            self.conv_seg = original_model.conv_seg

        # Pooling globale 3D + testa lineare
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # N×C×D×H×W -> N×C×1×1×1 [web:155]
        out_channels = self._infer_out_channels_from_layer4()  # C finale [web:172]
        self.fc = nn.Linear(out_channels, num_classes)         # N×C -> N×K [web:172]

    def _infer_out_channels_from_layer4(self) -> int:
        last = self.layer4[-1]
        if hasattr(last, "bn3"):  # Bottleneck(expansion=4)
            return last.bn3.num_features
        if hasattr(last, "bn2"):  # BasicBlock(expansion=1)
            return last.bn2.num_features
        raise RuntimeError("Impossibile inferire i canali di uscita da layer4; verifica la struttura dei blocchi.")  # [web:172]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Path backbone puro fino a layer4
        x = self.conv1(x)     # [B,C,D,H,W] -> conv [web:172]
        x = self.bn1(x)       # BN [web:172]
        x = self.relu(x)      # ReLU [web:172]
        x = self.maxpool(x)   # downsample [web:172]
        x = self.layer1(x)    # stadi residui [web:172]
        x = self.layer2(x)    # [web:172]
        x2 = self.layer3(x)    # [web:172]
        x = self.layer4(x2)    # feature map finale C_out [web:172]
        return x,x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)               # N×C×D×H×W [web:172]
        pooled = self.global_pool(feats)              # N×C×1×1×1 [web:155]
        flat = torch.flatten(pooled, 1)               # N×C [web:337]
        logits = self.fc(flat)                        # N×num_classes [web:172]
        return logits

    @torch.no_grad()
    def get_feature_dimensions(self, input_shape, device=None):
        """
        input_shape: (N,C,D,H,W) oppure (C,D,H,W)
        Ritorna (shape_feat, shape_pooled, shape_flat) per verifica rapida.
        """
        if device is None:
            device = next(self.parameters()).device
        if len(input_shape) == 5:
            dummy = torch.randn(*input_shape, device=device)
        elif len(input_shape) == 4:
            dummy = torch.randn(1, *input_shape, device=device)
        else:
            raise ValueError("input_shape deve essere (N,C,D,H,W) o (C,D,H,W)")  # [web:172]

        feats = self.forward_features(dummy)          # [web:172]
        pooled = self.global_pool(feats)              # [web:155]
        flat = torch.flatten(pooled, 1)               # [web:337]
        return feats.shape, pooled.shape, flat.shape

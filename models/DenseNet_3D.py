from monai.networks.nets import DenseNet201, DenseNet264
import torch
import torch.nn as nn

class DenseNet_3D(nn.Module):
    def __init__(self, original_model: nn.Module, num_classes: int = 3, num_features: int = 2048):
        super().__init__()
        # Copia backbone 3D
        self.features   = original_model.features

        # Mantieni la testa di segmentazione solo se ti serve altrove (non usata per la classificazione)
        self.has_seg_head = hasattr(original_model, "conv_seg")
        if self.has_seg_head:
            self.conv_seg = original_model.conv_seg

        # Pooling globale 3D + testa lineare
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # N×C×D×H×W -> N×C×1×1×1 [web:155]
        out_channels = self._infer_out_channels_from_layer4()  # C finale [web:172]
        self.fc = nn.Linear(out_channels, num_classes)         # N×C -> N×K [web:172]

    def _infer_out_channels_from_layer4(self) -> int:
        last = self.features[-1]
        print(last)

        # Caso DenseNet: l'ultimo layer di 'features' è una BatchNorm (es. norm5)
        if hasattr(last, "num_features"):
            return int(last.num_features)

        # Se l'ultimo è un blocco, prova attributi comuni
        for attr in ("norm5", "bn3", "bn2", "bn1"):
            if hasattr(last, attr):
                m = getattr(last, attr)
                if hasattr(m, "num_features"):
                    return int(m.num_features)

        # Fallback robusto: cerca l'ultimo modulo con canali noti
        for m in reversed(list(self.features.modules())):
            if hasattr(m, "num_features"):        # BatchNormNd
                return int(m.num_features)
            if hasattr(m, "out_channels"):        # ConvNd
                return int(m.out_channels)

        raise RuntimeError(
            "Impossibile inferire i canali di uscita; verifica la struttura dei blocchi."
        )


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Path backbone puro fino a layer4
        x = self.features(x)     # [B,C,D,H,W] -> conv [web:172]
        return x,x

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



if __name__ == "__main__":
    model = DenseNet201(spatial_dims=3, in_channels=1, out_channels=1)   
    print(model)
    train_params = 0
    for name, param in model.named_parameters():
        #print(name, param.shape)
        train_params += param.numel()
    print(f"Total trainable parameters: {train_params}")
    model_wrapper = DenseNet_3D(original_model=model, num_classes=3)
    print(model_wrapper)
    input_shape = (1, 1,64, 64, 64)
    feats,_ = model_wrapper.forward_features(torch.randn(input_shape))
    print("Feature shape:", feats.shape)
    pooled = model_wrapper.global_pool(feats)
    print("Pooled shape:", pooled.shape)
    flat = torch.flatten(pooled, 1)
    print("Flat shape:", flat.shape)
    logits = model_wrapper.fc(flat)
    print("Logits shape:", logits.shape)
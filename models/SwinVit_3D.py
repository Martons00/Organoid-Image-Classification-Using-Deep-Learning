import torch
import torch.nn as nn

class SwinVit_3D(nn.Module):
    def __init__(self, original_model, num_classes=3, num_features=384):
        super(SwinVit_3D, self).__init__()
        
        # Copiamo le componenti dell'encoder dal modello originale
        self.swinViT = original_model.swinViT
        
        # Attributi necessari per il forward del SwinViT
        self.normalize = getattr(original_model, 'normalize', True)
        
        # Attributi richiesti da add_ml_decoder_head
        self.num_classes = num_classes
        self.num_features = num_features
        
        # Pattern ResNet50: global_pool + fc (ADATTATO PER 3D)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Pool globale 3D per volumi
        self.fc = nn.Linear(num_features, num_classes)  # Testa di classificazione
        
    def forward_features(self, x):
        """Estrae le feature senza classificazione"""
        # Forward pass attraverso il Swin Transformer
        hidden_states_out = self.swinViT(x, self.normalize)
        # enc_hidden dovrebbe avere forma [B, 384, D, H, W]
        return hidden_states_out[4], hidden_states_out[3]
        
    def forward_all_features(self, x):
        """Ritorna tutte le feature per backward compatibility"""
        # Forward pass attraverso il Swin Transformer
        hidden_states_out = self.swinViT(x, self.normalize)
        
        return {
            'enc_hidden': hidden_states_out[4],
            'hidden_states': hidden_states_out
        }
        
    def forward(self, x):
        """Forward standard per classificazione compatibile con dati 3D"""
        # Estrai le feature: [B, C, D, H, W]
        features = self.forward_features(x)
        print(f"Features shape: {features.shape}")
        
        # Applica global pooling 3D se presente
        if hasattr(self, 'global_pool') and self.global_pool is not None:
            # features: [B, 384, D, H, W] -> [B, 384, 1, 1, 1]
            features = self.global_pool(features)
            
            # Flatten per la testa lineare: [B, 384, 1, 1, 1] -> [B, 384]
            features = features.flatten(1)
        
        # Applica la testa di classificazione
        print(f"Features shape before classifier: {features.shape}")
        if hasattr(self, 'fc'):
            return self.fc(features)  # [B, 384] -> [B, num_classes]
        elif hasattr(self, 'head'):
            return self.head(features)
        else:
            return features

    def get_feature_dimensions(self, input_shape):
        """Utility per verificare le dimensioni delle feature"""
        print(f"Input shape: {input_shape}")
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape[1:])  # Rimuovi batch dimension
            features = self.forward_features(dummy_input)
            pooled = self.global_pool(features) if hasattr(self, 'global_pool') else features
            flattened = pooled.flatten(1) if len(pooled.shape) > 2 else pooled
            print(f"Feature shape dopo encoder: {features.shape}")
            print(f"Shape dopo global pooling: {pooled.shape}")
            print(f"Shape dopo flatten: {flattened.shape}")
            return features.shape, pooled.shape, flattened.shape
        
if __name__ == "__main__":
    from monai.networks.nets import SwinUNETR
    
    # Crea il modello originale per volumi 3D
    original_model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=1,          # Canali input (es. modalità MRI)
        out_channels=3,         # Classi output
    )
    
    print("Modello originale SwinUNETR:")
    print(original_model)
    
    # Crea l'encoder adattato
    encoder_model = SwinVit_3D(
        original_model, 
        num_classes=3, 
        num_features=384  # Verifica che questo corrisponda all'output dell'encoder10
    )
    
    print("\nModello encoder adattato:")
    print(encoder_model)
    
 
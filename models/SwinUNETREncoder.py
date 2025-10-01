import torch
import torch.nn as nn

import torch
import torch.nn as nn


class SwinUNETREncoder(nn.Module):
    def __init__(self, original_model, num_classes=3, num_features=768):
        super(SwinUNETREncoder, self).__init__()
        
        # Copiamo le componenti dell'encoder dal modello originale
        self.swinViT = original_model.swinViT
        self.encoder1 = original_model.encoder1
        self.encoder2 = original_model.encoder2 
        self.encoder3 = original_model.encoder3
        self.encoder4 = original_model.encoder4
        self.encoder10 = original_model.encoder10
        
        # Attributi necessari per il forward del SwinViT
        self.normalize = getattr(original_model, 'normalize', True)
        
        # Attributi richiesti da add_ml_decoder_head
        self.num_classes = num_classes
        self.num_features = num_features
        
        # Pattern ResNet50: global_pool + fc
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Pool globale per ridimensionare
        self.fc = nn.Linear(num_features, num_classes)  # Testa di classificazione temporanea
        
        # Alternative: Pattern TResNet (commentato, usa solo uno dei due pattern)
        # self.head = nn.Linear(num_features, num_classes)
        # self.global_pool = nn.Identity()  # Opzionale per pattern TResNet
        
    def forward_features(self, x):
        """Estrae le feature senza classificazione"""
        # Forward pass attraverso il Swin Transformer
        hidden_states_out = self.swinViT(x, self.normalize)
        
        # Usa solo le feature finali dal layer più profondo
        enc_hidden = self.encoder10(hidden_states_out[4])
        
        return enc_hidden
        
    def forward_all_features(self, x):
        """Ritorna tutte le feature per backward compatibility"""
        # Forward pass attraverso il Swin Transformer
        hidden_states_out = self.swinViT(x, self.normalize)
        
        # Applica i blocchi encoder in sequenza
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(hidden_states_out[0])
        enc3 = self.encoder3(hidden_states_out[1]) 
        enc4 = self.encoder4(hidden_states_out[2])
        enc_hidden = self.encoder10(hidden_states_out[4])
        
        return {
            'enc1': enc1,
            'enc2': enc2,
            'enc3': enc3,  
            'enc4': enc4,
            'enc_hidden': enc_hidden,
            'hidden_states': hidden_states_out
        }
        
    def forward(self, x):
        """Forward standard per classificazione compatibile con ML-Decoder"""
        # Estrai le feature
        features = self.forward_features(x)
        
        # Applica global pooling se presente
        if hasattr(self, 'global_pool') and self.global_pool is not None:
            # Assicurati che le feature abbiano la forma corretta per pooling
            if len(features.shape) == 3:  # [B, N, C] -> [B, C, H, W]
                B, N, C = features.shape
                H = W = int(N ** 0.5)  # Assumi feature quadrate
                features = features.transpose(1, 2).reshape(B, C, H, W)
            
            features = self.global_pool(features)
            
            # Flatten per la testa lineare
            if len(features.shape) > 2:
                features = features.flatten(1)
        
        # Applica la testa di classificazione
        if hasattr(self, 'fc'):
            return self.fc(features)
        elif hasattr(self, 'head'):
            return self.head(features)
        else:
            return features



if __name__ == "__main__":
    from monai.networks.nets import SwinUNETR
    from ml_decoder.ml_decoder import add_ml_decoder_head
    
    # Crea il modello originale
    original_model = SwinUNETR(4, 3)
    
    # Crea l'encoder con attributi già compatibili
    encoder_model = SwinUNETREncoder(
        original_model, 
        num_classes=3, 
        num_features=768
    )
    
    print("Modello prima di ML-Decoder:")
    print(f"Ha global_pool: {hasattr(encoder_model, 'global_pool')}")
    print(f"Ha fc: {hasattr(encoder_model, 'fc')}")
    print(f"num_features: {encoder_model.num_features}")
    print(f"num_classes: {encoder_model.num_classes}")
    
    # Applica ML-Decoder (sostituirà automaticamente fc con MLDecoder)
    encoder_model = add_ml_decoder_head(encoder_model, num_classes=3)
    
    print("\nModello dopo ML-Decoder:")
    print(f"Tipo di fc: {type(encoder_model.fc)}")
    
    # Test forward
    test_input = torch.randn(1, 4, 96, 96, 96)  # Esempio input 3D
    output = encoder_model(test_input)
    print(f"Output shape: {output.shape}")

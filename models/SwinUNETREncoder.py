import torch
import torch.nn as nn

class SwinUNETREncoder(nn.Module):
    def __init__(self, original_model):
        super(SwinUNETREncoder, self).__init__()
        
        # Copiamo le componenti dell'encoder dal modello originale
        # Questi mantengono automaticamente i pesi pre-addestrati
        self.swinViT = original_model.swinViT
        self.encoder1 = original_model.encoder1
        self.encoder2 = original_model.encoder2 
        self.encoder3 = original_model.encoder3
        self.encoder4 = original_model.encoder4
        self.encoder10 = original_model.encoder10
        
        # Attributi necessari per il forward del SwinViT
        self.normalize = getattr(original_model, 'normalize', True)
        
    def forward(self, x):
        # Forward pass attraverso il Swin Transformer
        hidden_states_out = self.swinViT(x, self.normalize)
        
        # Applica i blocchi encoder in sequenza
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(hidden_states_out[0])
        enc3 = self.encoder3(hidden_states_out[1]) 
        enc4 = self.encoder4(hidden_states_out[2])
        enc_hidden = self.encoder10(hidden_states_out[4])
        
        # Ritorna le features encodate a diverse risoluzioni
        return {
            'enc1': enc1,           # Features originali processate
            'enc2': enc2,           # Features da layer 1
            'enc3': enc3,           # Features da layer 2  
            'enc4': enc4,           # Features da layer 3
            'enc_hidden': enc_hidden,  # Features finali dal layer 4
            'hidden_states': hidden_states_out
        }


if __name__ == "__main__":
    # Esempio di utilizzo
    from monai.networks.nets import SwinUNETR
    model = SwinUNETR(4, 3)  # Modello originale
    print(model)
    new_model = SwinUNETREncoder(model)  # Modello con solo l'encoder
    print(new_model)
    new_model.fc = nn.Identity()  # Rimuoviamo la testa di classificazione
    new_model.num_features = 768  # Aggiungiamo l'attributo num_features
    from ML_Decoder_main.src_files.ml_decoder.ml_decoder import add_ml_decoder_head

    new_model = add_ml_decoder_head(new_model, num_classes=3)
    print(new_model)
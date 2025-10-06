
import torch
import numpy as np
import tifffile
import os
import sys
import argparse
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset
from SwinUNETREncoder_3D import SwinUNETREncoder


def load_ml_decoder():
    """Carica ML-Decoder se disponibile."""
    try:
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
        SRC = os.path.join(ROOT, 'models', 'ML_Decoder_main', 'src_files')

        if SRC not in sys.path:
            sys.path.insert(0, SRC)

        from ml_decoder.ml_decoder import MLDecoder
        return MLDecoder
    except ImportError:
        print("ML-Decoder non disponibile")
        return None

def add_ml_decoder():
    """Carica ML-Decoder se disponibile."""
    try:
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
        SRC = os.path.join(ROOT, 'models', 'ML_Decoder_main', 'src_files')

        if SRC not in sys.path:
            sys.path.insert(0, SRC)

        from ml_decoder.ml_decoder import add_ml_decoder_head
        return add_ml_decoder_head
    except ImportError:
        print("ML-Decoder non disponibile")
        return None


def create_model(model_path, device, use_ml_decoder=True):
    """Crea il modello SwinUNETR con encoder personalizzato."""
    # Modello base
    model = SwinUNETR(
        img_size=(128, 128, 128), 
        in_channels=1, 
        out_channels=1, 
        feature_size=48,
        use_checkpoint=True
    )
    model.eval()

    # Carica i pesi
    weights = torch.load(model_path, weights_only=True, map_location=device)
    model.load_from(weights=weights)
    print("Pesi pre-addestrati caricati con successo")

    # Crea l'encoder personalizzato
    encoder_model = SwinUNETREncoder(
        model, 
        num_classes=3, 
        num_features=768
    )

    # Applica ML-Decoder se richiesto e disponibile
    if use_ml_decoder:
        '''
        MLDecoder = load_ml_decoder()
        if MLDecoder:
            head = MLDecoder(
                num_classes=3, 
                initial_num_features=768, 
                num_of_groups=1, 
                decoder_embedding=768, 
                zsl=0
            )
            encoder_model.global_pool = torch.nn.Identity()
            encoder_model.fc = head
            print("ML-Decoder applicato con successo")
        '''
        add_ml_decoder_head = add_ml_decoder()
        if add_ml_decoder_head:
            encoder_model = add_ml_decoder_head(encoder_model, num_classes=3)
            print("ML-Decoder applicato con successo")
            print (f"Modello con ML-Decoder:\n{encoder_model}")

    return encoder_model.to(device)


def load_and_preprocess_image(image_path, target_size=(128, 128, 128)):
    """Carica e preprocessa l'immagine."""
    data = tifffile.imread(image_path)
    print(f"Immagine originale: {data.shape}, dtype: {data.dtype}")

    # Crop alle dimensioni del modello
    data = data[:target_size[0], :target_size[1], :target_size[2]]

    # Aggiungi dimensione canale: (D, H, W) -> (1, D, H, W)
    data = np.expand_dims(data, axis=0)
    print(f"Immagine preprocessata: {data.shape}")

    return data


def run_inference(model, data, device):
    """Esegue l'inferenza sul modello."""
    dataset = Dataset([data])
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    results = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device).to(torch.float32)
            print(f"Input shape: {inputs.shape}")

            # Inferenza
            output = model(inputs)
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

            results.append(output.cpu())

    return results


def main():
    parser = argparse.ArgumentParser(description="Inferenza Swin UNETR con encoder personalizzato")
    parser.add_argument("image_path", help="Percorso al file immagine")
    parser.add_argument("--model_path", default="./model_swinvit.pt", help="Percorso al modello")
    parser.add_argument("--output_dir", default="output", help="Directory di output")
    parser.add_argument("--no_ml_decoder", action="store_true", help="Disabilita ML-Decoder")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Carica modello
    model = create_model(args.model_path, device, use_ml_decoder=not args.no_ml_decoder)

    # Carica e preprocessa immagine
    data = load_and_preprocess_image(args.image_path)

    # Inferenza
    print("Esecuzione inferenza...")
    results = run_inference(model, data, device)

    print(f"Inferenza completata. Risultati: {len(results)} batch")

    # Salvataggio opzionale (se necessario)
    # for i, result in enumerate(results):
    #     output_path = os.path.join(args.output_dir, f"result_{i}.tif")
    #     # Salva risultato...

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

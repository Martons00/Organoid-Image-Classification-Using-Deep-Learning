import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    ToTensord,
    CropForegroundd,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    EnsureTyped,
    RepeatChanneld,
    AddChanneld
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
import nibabel as nib
import argparse
import os

def create_transforms_for_tif():
    """Crea trasformazioni ottimizzate per file .tif"""
    transforms = Compose([
        # Carica con reader specifico per TIFF
        LoadImaged(keys=["image"], reader="ITKReader"),

        # Assicura che ci sia una dimensione canale
        EnsureChannelFirstd(keys=["image"]),

        # Se è un'immagine singolo canale, duplica per simulare multimodale
        # Nota: questo è un workaround, non ideale per performance reali
        RepeatChanneld(keys=["image"], repeats=4),  # Simula T1, T1ce, T2, FLAIR

        # Cerca di applicare orientazione standard (potrebbe fallire senza metadati)
        # Orientationd(keys=["image"], axcodes="RAS"),  # Commentato per sicurezza

        # Resample se possibile (potrebbe fallire senza spacing info)
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),

        # Crop foreground se c'è contrasto sufficiente
        CropForegroundd(keys=["image"], source_key="image"),

        # Resize forzato alle dimensioni del modello
        ResizeWithPadOrCropd(keys=["image"], spatial_size=[128, 128, 128]),

        # Normalizzazione
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Conversione a tensor
        EnsureTyped(keys=["image"]),
    ])
    return transforms

def create_transforms_standard():
    """Trasformazioni standard per file NIfTI (come nel codice originale)"""
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=[128, 128, 128]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])
    return transforms

def load_model(model_path, device):
    """Carica il modello Swin UNETR"""
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Modello caricato da {model_path}")
    else:
        print(f"Attenzione: File modello {model_path} non trovato, usando pesi random")

    model.to(device)
    model.eval()
    return model

def run_inference(image_path, model_path, output_dir="output"): 
    """Esegue l'inferenza su una singola immagine con gestione intelligente del formato"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)

    # Carica il modello
    model = load_model(model_path, device)

    # Prepara i dati
    data_dict = {"image": image_path}

    # Determina il tipo di trasformazioni in base all'estensione del file
    file_ext = os.path.splitext(image_path.lower())[1]
    is_tif = file_ext in ['.tif', '.tiff']

    if is_tif:
        print("File .tif rilevato - usando trasformazioni ottimizzate per TIFF")
        transforms = create_transforms_for_tif()
    else:
        print("File medico standard rilevato - usando trasformazioni standard")
        transforms = create_transforms_standard()

    try:
        # Applica le trasformazioni
        print("Applicazione trasformazioni...")
        data = transforms(data_dict)

        print(f"Shape dopo trasformazioni: {data['image'].shape}")

        # Crea dataset e dataloader
        dataset = Dataset([data], transform=None)
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(device)

                print(f"Input shape al modello: {inputs.shape}")

                # Verifica che l'input abbia la forma corretta
                if inputs.dim() != 5 or inputs.shape[1] != 4:
                    raise ValueError(f"Input shape non corretto: {inputs.shape}. Atteso: [B, 4, H, W, D]")

                # Inferenza con sliding window
                print("Esecuzione inferenza...")
                outputs = sliding_window_inference(
                    inputs, 
                    roi_size=(128, 128, 128),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.6
                )

                # Post-processing
                outputs = torch.softmax(outputs, 1).cpu().numpy()
                outputs = np.argmax(outputs, axis=1).astype(np.uint8)[0]

                # Salva il risultato
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = os.path.join(output_dir, f"segmented_{base_name}.nii.gz")

                # Salva come file NIfTI
                img = nib.Nifti1Image(outputs, np.eye(4))
                nib.save(img, output_filename)

                print(f"Segmentazione salvata in: {output_filename}")

                # Salva statistiche
                stats_file = os.path.join(output_dir, f"segmentation_stats_{base_name}.txt")
                with open(stats_file, "w") as f:
                    f.write(f"File input: {image_path}\n")
                    f.write(f"Formato file: {file_ext}\n")
                    f.write(f"Dimensioni output: {outputs.shape}\n")
                    f.write(f"Classi uniche trovate: {np.unique(outputs)}\n")
                    f.write(f"Numero voxel per classe:\n")
                    for class_id in np.unique(outputs):
                        count = np.sum(outputs == class_id)
                        percentage = (count / outputs.size) * 100
                        f.write(f"  Classe {class_id}: {count} voxel ({percentage:.2f}%)\n")

                    if is_tif:
                        f.write(f"\nAVVISO: File .tif processato con workaround\n")
                        f.write(f"- Canali duplicati da 1 a 4 per compatibilità\n")
                        f.write(f"- Metadati spaziali potrebbero essere inaccurati\n")
                        f.write(f"- Risultati da validare attentamente\n")

                print(f"Statistiche salvate in: {stats_file}")

                return True

    except Exception as e:
        print(f"Errore durante l'inferenza con MONAI: {str(e)}")

        # Fallback per caricamento diretto
        print("Tentativo di caricamento diretto del file...")
        try:
            if is_tif:
                import tifffile
                img_array = tifffile.imread(image_path)
            else:
                import nibabel as nib
                img_obj = nib.load(image_path)
                img_array = img_obj.get_fdata()

            print(f"Immagine caricata con dimensioni: {img_array.shape}")

            # Salva informazioni diagnostiche
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            info_file = os.path.join(output_dir, f"diagnostic_info_{base_name}.txt")
            with open(info_file, "w") as f:
                f.write(f"File input: {image_path}\n")
                f.write(f"Formato: {file_ext}\n")
                f.write(f"Dimensioni: {img_array.shape}\n")
                f.write(f"Tipo dati: {img_array.dtype}\n")
                f.write(f"Min value: {img_array.min()}\n")
                f.write(f"Max value: {img_array.max()}\n")
                f.write(f"\nERRORE DURANTE ELABORAZIONE:\n{str(e)}\n")
                f.write(f"\nRECOMANDAZIONI:\n")
                f.write(f"- Convertire il file in formato NIfTI (.nii.gz)\n")
                f.write(f"- Verificare che sia un'immagine 3D medica\n")
                f.write(f"- Assicurarsi che abbia 4 modalità (T1, T1ce, T2, FLAIR)\n")

            print(f"Informazioni diagnostiche salvate in: {info_file}")
            return False

        except Exception as e2:
            print(f"Errore anche nel caricamento diretto: {str(e2)}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferenza Swin UNETR con supporto migliorato per .tif")
    parser.add_argument("image_path", help="Percorso al file immagine (.tif, .nii.gz, etc.)")
    parser.add_argument("--model_path", default="pretrained_models/fold1_f48_ep300_4gpu_dice0_9059.pth", 
                       help="Percorso al modello pre-trained")
    parser.add_argument("--output_dir", default="output", help="Directory di output")

    args = parser.parse_args()

    success = run_inference(args.image_path, args.model_path, args.output_dir)

    if success:
        print("\n✅ Inferenza completata con successo!")
    else:
        print("\n❌ Inferenza fallita - controllare i file diagnostici per dettagli")
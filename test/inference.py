import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd,
    Orientationd, Spacingd, EnsureTyped
)
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
import argparse
import os
import sys
from SwinUNETREncoder_3D import SwinUNETREncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferenza Swin UNETR senza funzioni (main-only)")
    parser.add_argument("image_path", help="Percorso al file immagine (.tif, .tiff, .nii.gz, ecc.)")
    parser.add_argument("--model_path", default="./model_swinvit.pt", help="Percorso al modello pre-trained")
    parser.add_argument("--output_dir", default="output", help="Directory di output")
    args = parser.parse_args()

    image_pth = args.image_path
    model_pth = args.model_path
    output_dir = args.output_dir

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")



    # Output dir
    os.makedirs(output_dir, exist_ok=True)

    # Modello
    model = SwinUNETR(img_size=(128,512,512) , in_channels=1, out_channels=1, feature_size=48,use_checkpoint=True)
    model.eval()
    weight = torch.load(model_pth, weights_only=True, map_location=device if torch.cuda.is_available() else "cpu")
    model.load_from(weights=weight)
    print("Using pretrained self-supervied Swin UNETR backbone weights !")
        # Crea l'encoder adattato
    encoder_model = SwinUNETREncoder(
        model, 
        num_classes=3, 
        num_features=768  # Verifica che questo corrisponda all'output dell'encoder10
    )
    
    print("\nModello encoder adattato:")
    print(f"Ha global_pool (3D): {hasattr(encoder_model, 'global_pool')}")
    print(f"Tipo global_pool: {type(encoder_model.global_pool)}")
    print(f"Ha fc: {hasattr(encoder_model, 'fc')}")
    print(f"num_features: {encoder_model.num_features}")
    print(f"num_classes: {encoder_model.num_classes}")
    print(encoder_model)
    model = encoder_model.to(device)

    import tifffile
    data = tifffile.imread(image_pth)
    print(f"Input image shape: {data.shape}, dtype: {data.dtype}")
    data = data[:128,:512,:512]  # Crop to fit model input size
    data = np.expand_dims(data, axis=0)  # da (D, H, W) a (1, D, H, W)

    print(f"Cropped image shape: {data.shape}, dtype: {data.dtype}")
    img_np = data  # shape: (165, 512, 512), dtype=uint8
    lbl_np = None  # shape: (165, 512, 512) o None se assente

    # Spaziatura nota o assunta (sx, sy, sz) in mm
    sx, sy, sz = 1.0, 1.0, 1.0
    affine = np.diag([sx, sy, sz, 1.0])

    sample = {
        "image": img_np,  # (D, H, W)
        "label": lbl_np,  # opzionale
        "image_meta_dict": {"affine": affine},
        "label_meta_dict": {"affine": affine},
    }

    dataset = Dataset([data], transform=None)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device)
            
            inputs = inputs.to(torch.float32)
            print(f"Input shape al modello: {inputs.shape}")

            print("Esecuzione inferenza...")
            output = model(inputs)
            #output = model.forward_features(inputs)
            print(f"Output shape dal modello: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            print(f"Output min/max: {output.min().item()}/{output.max().item()}")

            '''
            pred = output.argmax(dim=1)     # [1, 128, 512, 512]
            pred = pred.squeeze(0)          # [128, 512, 512]

            pred_np = pred.detach().cpu().numpy().astype(np.uint8)




            # NOTA: salvataggio disabilitato (nessuna dipendenza nibabel/SimpleITK qui)
            print(pred.shape)
            print(f"Segmentazione salvata in: {output_dir}")

            tifffile.imwrite(str(output_dir+"text.tif"), pred_np)
            '''

            success = True
            break

        sys.exit(0 if success else 1)

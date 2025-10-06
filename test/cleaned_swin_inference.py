
import torch
import numpy as np
import tifffile
import os
import sys
import argparse
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset
from SwinUNETREncoder_3D import  SwinUNETREncoder_only
from SwinUnetr_test import SwinUNETREncoder
from monai.inferers import  sliding_window_inference
import torch.nn.functional as F

import numpy as np

import torch

# feats: [N, 768, fD, fH, fW] in ordine (z major -> y -> x) come nell’estrazione
def tile_feature_patches(feats: torch.Tensor, coords) -> torch.Tensor:
    # feats: [N, C, fD, fH, fW] dove N = nZ*nY*nX patch in ordine z->y->x
        # Calcola la griglia dalle coordinate
    unique_coords = {}
    for i, (b, z, y, x) in enumerate(coords):
        if b not in unique_coords:
            unique_coords[b] = {'z': set(), 'y': set(), 'x': set()}
        unique_coords[b]['z'].add(z)
        unique_coords[b]['y'].add(y)  
        unique_coords[b]['x'].add(x)
    
    # Assumendo batch=0
    nZ = len(unique_coords[0]['z'])
    nY = len(unique_coords[0]['y']) 
    nX = len(unique_coords[0]['x'])
    print(f"Griglia ricomposta: nZ={nZ}, nY={nY}, nX={nX}")

    N, C, fD, fH, fW = feats.shape
    assert C == 768, f"Attesi 768 canali, trovato {C}"
    assert N == nZ * nY * nX, f"N non combacia con griglia: N={N} vs {nZ*nY*nX}"
    
    # [N, C, fD, fH, fW] -> [nZ, nY, nX, C, fD, fH, fW]
    g = feats.reshape(nZ, nY, nX, C, fD, fH, fW)
    
    # Riordina: [nZ, nY, nX, C, fD, fH, fW] -> [C, nZ, fD, nY, fH, nX, fW]  
    g = g.permute(3, 0, 4, 1, 5, 2, 6).contiguous()
    
    # Ricomponi le dimensioni spaziali: [C, nZ*fD, nY*fH, nX*fW]
    vol = g.reshape(C, nZ*fD, nY*fH, nX*fW)
    
    # Aggiungi dimensione batch: [1, C, nZ*fD, nY*fH, nX*fW]
    return vol.unsqueeze(0)


def ensure_single_channel(x, mode="first"):
    # x: [B,C,D,H,W] oppure [B,D,H,W]
    if x.dim() == 4:
        x = x.unsqueeze(1)  # -> [B,1,D,H,W]
    if x.shape[1] != 1:
        if mode == "first":
            x = x[:, :1]  # usa il primo canale/slice
        elif mode == "mean":
            x = x.mean(dim=1, keepdim=True)  # media sui canali
        else:
            raise ValueError("mode deve essere 'first' o 'mean'")
    return x

def _starts(size, patch, step):
    if size <= patch:
        return [0]
    s = list(range(0, size - patch + 1, step))
    if s[-1] != size - patch:
        s.append(size - patch)
    return s

def extract_patches_5d_torch(x, patch_size=(128,256,256), step=(128,256,256), pad_value=0):
    # x: [B,1,D,H,W], ritorna patches: [N,1,pd,ph,pw] e coords: [(b,z,y,x0), ...]
    B, C, D, H, W = x.shape
    pd, ph, pw = patch_size
    sd, sh, sw = step
    zs, ys, xs = _starts(D, pd, sd), _starts(H, ph, sh), _starts(W, pw, sw)

    patches = []
    coords  = []
    for b in range(B):
        for z in zs:
            for y in ys:
                for x0 in xs:
                    patch = x[b:b+1, :, z:z+pd, y:y+ph, x0:x0+pw]  # [1,1,d',h',w']
                    dd, hh, ww = patch.shape[-3:]
                    if (dd, hh, ww) != (pd, ph, pw):
                        # pad solo a destra su D,H,W: (wL,wR,hL,hR,dL,dR)
                        pad_d = pd - dd
                        pad_h = ph - hh
                        pad_w = pw - ww
                        patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), value=pad_value)
                    patches.append(patch)   # [1,1,pd,ph,pw]
                    coords.append((b, z, y, x0))
    if not patches:
        return torch.empty(0, 1, *patch_size), []
    patches = torch.cat(patches, dim=0)  # [N,1,pd,ph,pw]
    return patches, coords



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
        MLDecoder = load_ml_decoder()
        if MLDecoder:
            head = MLDecoder(
                num_classes=3, 
                initial_num_features=1024, 
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

        '''
    return encoder_model.to(device)

def create_encoder(model_path, device, use_ml_decoder=True):
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
    encoder_model = SwinUNETREncoder_only(
        model, 
        num_features=768
    )

    # Applica ML-Decoder se richiesto e disponibile
    if use_ml_decoder:
        MLDecoder = load_ml_decoder()
        if MLDecoder:
            head = MLDecoder(
                num_classes=3, 
                initial_num_features=1024, 
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

        '''
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
            inputs = batch
            # Uso nel tuo codice:
            print(f"Input shape: {inputs.shape}")  # atteso [B,C,D,H,W] o [B,D,H,W]

            inputs = ensure_single_channel(inputs, mode="first")  # -> [B,1,D,H,W]
            patches, coords = extract_patches_5d_torch(
                inputs, patch_size=(128,128,128), step=(128,128,128), pad_value=0
            )
            print(f"Patches shape: {patches.shape}")  # atteso [N,1,128,128,128] 
            # batch inferenza
            results = []
            for i, patch in enumerate(patches):
                patch = patches[i:i+1]  # [1,1,128,128,128]
                patch = patch.to(device).to(torch.float32)  # [1,1,128,128,128]
                print(f"Patch {i+1}/{len(patches)} shape: {patch.shape}, dtype: {patch.dtype}")
                output = model.forward_features(patch)    # [1,3,1,1,3] o [1,3] con ML-Decoder
                print(f"Patch {i+1}/{len(patches)} inferita. Output shape: {output.shape}")
                print(f"Patch {i+1}/{len(patches)} inferita. coords: {coords[i]}")
                results.append(output)
            output = torch.cat(results, dim=0)  # [N,3] con ML-Decoder
            output = tile_feature_patches(output, coords=coords)  # [1,3,1,1,N]
            print(f"Features shape: {output.shape}")
            print(f"Features range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            # Classificazione finale

            output = model.global_pool(output)  # [1,3]
            print(f"Output dopo global_pool shape: {output.shape}") 
            output = output.flatten(1) 
            outputs = model.fc(output)  
            print(f"Output finale shape: {outputs.shape}")
            print(f"Output finale range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")


    return outputs
# Funzione semplificata per run_inference
def run_inference_new(model, data, device):
    """Inferenza semplificata - il modello gestisce tutto internamente"""
    dataset = Dataset([data])
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    results = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device).to(torch.float32)
            
            # Una singola chiamata - tutto è gestito internamente
            output = model(inputs)
            
            print(f"Output finale: {output.shape}")
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
    #model = create_encoder(args.model_path, device, use_ml_decoder=not args.no_ml_decoder)

    # Carica e preprocessa immagine
    data = load_and_preprocess_image(args.image_path, target_size=(128, 512, 512))
    # Iterator di sole patch (senza inferenza)



    # Inferenza
    print("Esecuzione inferenza...")
    #
    # results = run_inference(model, data, device)
    results = run_inference_new(model, data, device)

    print(f"Inferenza completata. Risultati: {len(results)} batch")

    # Salvataggio opzionale (se necessario)
    # for i, result in enumerate(results):
    #     output_path = os.path.join(args.output_dir, f"result_{i}.tif")
    #     # Salva risultato...

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

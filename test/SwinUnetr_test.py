import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinUNETREncoder(nn.Module):
    def __init__(self, original_model, num_classes=3, num_features=768, 
                 patch_size=(128,128,128), step=(128,128,128)):
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
        
        # Configurazione patch
        self.patch_size = patch_size
        self.step = step
        
        # Attributi richiesti
        self.num_classes = num_classes
        self.num_features = num_features
        
        # Pattern ResNet50: global_pool + fc 
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Sarà sostituito da Identity se ML-Decoder
        self.fc = nn.Linear(num_features, num_classes)  # Sarà sostituito da ML-Decoder
        
    def ensure_single_channel(self, x, mode="first"):
        """x: [B,C,D,H,W] o [B,D,H,W] -> [B,1,D,H,W]"""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.shape[1] != 1:
            x = x[:, :1] if mode == "first" else x.mean(dim=1, keepdim=True)
        return x

    def _starts(self, size, patch, step):
        if size <= patch:
            return [0]
        s = list(range(0, size - patch + 1, step))
        if s[-1] != size - patch:
            s.append(size - patch)
        return s

    def extract_patches_5d_torch(self, x, pad_value=0):
        """x: [B,1,D,H,W] -> patches: [N,1,pd,ph,pw], coords: [(b,z,y,x0), ...]"""
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.step
        zs, ys, xs = self._starts(D, pd, sd), self._starts(H, ph, sh), self._starts(W, pw, sw)

        patches = []
        coords = []
        for b in range(B):
            for z in zs:
                for y in ys:
                    for x0 in xs:
                        patch = x[b:b+1, :, z:z+pd, y:y+ph, x0:x0+pw]  # [1,1,d',h',w']
                        dd, hh, ww = patch.shape[-3:]
                        if (dd, hh, ww) != (pd, ph, pw):
                            pad_d = pd - dd; pad_h = ph - hh; pad_w = pw - ww
                            patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), value=pad_value)
                        patches.append(patch)
                        coords.append((b, z, y, x0))
        
        if not patches:
            return torch.empty(0, 1, *self.patch_size, device=x.device), []
        patches = torch.cat(patches, dim=0)
        return patches, coords

    def tile_feature_patches(self, feats, coords):
        """Ricompone feature patches in volume finale"""
        if len(feats) == 0:
            return feats
            
        # Calcola griglia dalle coordinate
        unique_coords = {'z': set(), 'y': set(), 'x': set()}
        for (b, z, y, x) in coords:
            unique_coords['z'].add(z)
            unique_coords['y'].add(y)
            unique_coords['x'].add(x)
        
        nZ = len(unique_coords['z'])
        nY = len(unique_coords['y']) 
        nX = len(unique_coords['x'])
        
        N, C, fD, fH, fW = feats.shape
        assert N == nZ * nY * nX, f"N={N} vs griglia={nZ*nY*nX}"
        
        # Ricomponi
        g = feats.reshape(nZ, nY, nX, C, fD, fH, fW)
        g = g.permute(3, 0, 4, 1, 5, 2, 6).contiguous()
        vol = g.reshape(C, nZ*fD, nY*fH, nX*fW)
        return vol.unsqueeze(0)  # [1, C, ND, NH, NW]

    def forward_features_single_patch(self, x):
        """Estrae feature da una singola patch [1,1,D,H,W]"""
        hidden_states_out = self.swinViT(x, self.normalize)
        enc_hidden = self.encoder10(hidden_states_out[4])
        return enc_hidden
        
    def forward(self, x):
        """Forward completo con patch splitting interno"""
        print(f"Input shape: {x.shape}")
        
        # Normalizza a single channel
        x = self.ensure_single_channel(x, mode="first")
        
        # Estrai patch
        patches, coords = self.extract_patches_5d_torch(x, pad_value=0)
        print(f"Patches estratte: {patches.shape[0]}, shape singola: {tuple(patches.shape[1:])}")
        
        if len(patches) == 0:
            # Fallback per input molto piccolo
            features = self.forward_features_single_patch(x)
            features = self.global_pool(features)
            features = features.flatten(1)
            return self.fc(features)
        
        # Inferenza su ogni patch
        patch_features = []
        for i, patch in enumerate(patches):
            patch_single = patch.unsqueeze(0)  # [1,1,D,H,W]
            with torch.no_grad():
                feat = self.forward_features_single_patch(patch_single)  # [1,768,4,4,4]
            patch_features.append(feat)
            print(f"Patch {i+1}/{len(patches)} processata: {feat.shape}")
        
        # Ricomponi features
        all_features = torch.cat(patch_features, dim=0)  # [N,768,4,4,4]
        tiled_volume = self.tile_feature_patches(all_features, coords)  # [1,768,ND,NH,NW]
        print(f"Volume ricomposto: {tiled_volume.shape}")
        
        # Classificazione finale
        if isinstance(self.global_pool, nn.Identity):
            # ML-Decoder: converte a formato token
            B, C, D, H, W = tiled_volume.shape
            tokens = tiled_volume.flatten(2).transpose(1, 2)  # [B, N, C]
            return self.fc(tokens)  # ML-Decoder gestisce [B, N, 768]
        else:
            # Standard pooling + linear
            pooled = self.global_pool(tiled_volume)  # [1,768,1,1,1]
            pooled = pooled.flatten(1)  # [1,768]
            return self.fc(pooled)  # [1,num_classes]
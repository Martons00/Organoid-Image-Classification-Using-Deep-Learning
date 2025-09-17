import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import glob
import napari


def load_file_and_visualize_3d_planes(data_folder="../data"):
    """
    Carica un file .tif 3D e visualizza i tre piani ortogonali (xy, xz, yz)
    """
    
    # Trova tutti i file .tif nella cartella
    tif_files = glob.glob(os.path.join(data_folder))
    
    if not tif_files:
        print(f"Nessun file .tif trovato nella cartella {data_folder}")
        return
    
    # Usa il primo file trovato (o puoi modificare per scegliere)
    tif_path = tif_files[0]
    print(f"Caricando: {tif_path}")
    
    # Carica il volume 3D
    try:
        volume = io.imread(tif_path)
        print(f"Dimensioni volume: {volume.shape}")
        
        # Assicurati che sia 3D
        if len(volume.shape) < 3:
            print("Il file non sembra essere un volume 3D")
            return
            
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return
    
    # Se il volume ha 4 dimensioni (z, y, x, canali), prendi solo il primo canale
    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]
    
    z_size, y_size, x_size = volume.shape
    
    # Calcola le slice centrali per ogni piano
    z_center = z_size // 2
    y_center = y_size // 2
    x_center = x_size // 2
    
    # Estrai le slice dai tre piani
    xy_slice = volume[z_center, :, :]      # Piano xy (fissando z)
    xz_slice = volume[:, y_center, :]      # Piano xz (fissando y)
    yz_slice = volume[:, :, x_center]      # Piano yz (fissando x)
    
    # Visualizza i tre piani
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Piano XY (vista dall'alto)
    axes[0, 0].imshow(xy_slice, cmap='gray')
    axes[0, 0].set_title(f'Piano XY (z={z_center})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    # Piano XZ (sezione laterale)
    axes[0, 1].imshow(xz_slice, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Piano XZ (y={y_center})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    
    # Piano YZ (sezione frontale)
    axes[1, 0].imshow(yz_slice, cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'Piano YZ (x={x_center})')
    axes[1, 0].set_xlabel('Y')
    axes[1, 0].set_ylabel('Z')
    
    # Informazioni sul volume
    axes[1, 1].axis('off')
    info_text = f"""
    Volume Info:
    - Dimensioni: {volume.shape}
    - Tipo: {volume.dtype}
    - Min/Max: {volume.min()}/{volume.max()}
    - File: {os.path.basename(tif_path)}
    
    Piani visualizzati:
    - XY: slice centrale z={z_center}
    - XZ: slice centrale y={y_center}  
    - YZ: slice centrale x={x_center}
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    return volume

def load_and_visualize_3d_planes(data_folder="../data"):
    """
    Carica un file .tif 3D e visualizza i tre piani ortogonali (xy, xz, yz)
    """
    
    # Trova tutti i file .tif nella cartella
    tif_files = glob.glob(os.path.join(data_folder, "*.tif"))
    
    if not tif_files:
        print(f"Nessun file .tif trovato nella cartella {data_folder}")
        return
    
    # Usa il primo file trovato (o puoi modificare per scegliere)
    tif_path = tif_files[0]
    print(f"Caricando: {tif_path}")
    
    # Carica il volume 3D
    try:
        volume = io.imread(tif_path)
        print(f"Dimensioni volume: {volume.shape}")
        
        # Assicurati che sia 3D
        if len(volume.shape) < 3:
            print("Il file non sembra essere un volume 3D")
            return
            
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return
    
    # Se il volume ha 4 dimensioni (z, y, x, canali), prendi solo il primo canale
    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]
    
    z_size, y_size, x_size = volume.shape
    
    # Calcola le slice centrali per ogni piano
    z_center = z_size // 2
    y_center = y_size // 2
    x_center = x_size // 2
    
    # Estrai le slice dai tre piani
    xy_slice = volume[z_center, :, :]      # Piano xy (fissando z)
    xz_slice = volume[:, y_center, :]      # Piano xz (fissando y)
    yz_slice = volume[:, :, x_center]      # Piano yz (fissando x)
    
    # Visualizza i tre piani
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Piano XY (vista dall'alto)
    axes[0, 0].imshow(xy_slice, cmap='gray')
    axes[0, 0].set_title(f'Piano XY (z={z_center})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    # Piano XZ (sezione laterale)
    axes[0, 1].imshow(xz_slice, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Piano XZ (y={y_center})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    
    # Piano YZ (sezione frontale)
    axes[1, 0].imshow(yz_slice, cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'Piano YZ (x={x_center})')
    axes[1, 0].set_xlabel('Y')
    axes[1, 0].set_ylabel('Z')
    
    # Informazioni sul volume
    axes[1, 1].axis('off')
    info_text = f"""
    Volume Info:
    - Dimensioni: {volume.shape}
    - Tipo: {volume.dtype}
    - Min/Max: {volume.min()}/{volume.max()}
    - File: {os.path.basename(tif_path)}
    
    Piani visualizzati:
    - XY: slice centrale z={z_center}
    - XZ: slice centrale y={y_center}  
    - YZ: slice centrale x={x_center}
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    return volume

def interactive_slice_viewer(data_folder="../data"):
    """
    Versione interattiva per esplorare diverse slice
    """
    import ipywidgets as widgets
    from IPython.display import display
    
    # Carica il volume
    tif_files = glob.glob(os.path.join(data_folder, "*.tif"))
    if not tif_files:
        print(f"Nessun file .tif trovato nella cartella {data_folder}")
        return
        
    volume = io.imread(tif_files[0])
    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]
    
    z_size, y_size, x_size = volume.shape
    
    def plot_slices(z_idx, y_idx, x_idx):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Piano XY
        axes[0].imshow(volume[z_idx, :, :], cmap='gray')
        axes[0].set_title(f'Piano XY (z={z_idx})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Piano XZ  
        axes[1].imshow(volume[:, y_idx, :], cmap='gray', aspect='auto')
        axes[1].set_title(f'Piano XZ (y={y_idx})')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        
        # Piano YZ
        axes[2].imshow(volume[:, :, x_idx], cmap='gray', aspect='auto')
        axes[2].set_title(f'Piano YZ (x={x_idx})')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        
        plt.tight_layout()
        plt.show()
    
    # Widget interattivi
    z_slider = widgets.IntSlider(value=z_size//2, min=0, max=z_size-1, description='Z slice:')
    y_slider = widgets.IntSlider(value=y_size//2, min=0, max=y_size-1, description='Y slice:')
    x_slider = widgets.IntSlider(value=x_size//2, min=0, max=x_size-1, description='X slice:')
    
    widgets.interact(plot_slices, z_idx=z_slider, y_idx=y_slider, x_idx=x_slider)

#url = '/Volumes/LaCie/Noyaux/Chouxfleurs/raw_202502_Nice_orga0_31.tif'
#volume = load_file_and_visualize_3d_planes(url)
#url = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data/processed_cropped/processed_202407_Nice_orga0_5.tif"
url = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data/selected_z_ranges/202407_Nice_orga0_6_z120-220.tif"
#volume = load_and_visualize_3d_planes(url)
volume = load_file_and_visualize_3d_planes(url)

viewer = napari.Viewer()
viewer.add_image(volume, name="3D Volume", scale=[2, 1, 1])  # z, y, x scaling
# Avvia l'interfaccia interattiva
napari.run()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
from skimage.transform import resize
import tifffile as tiff
import os
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import datetime
import warnings

# Sopprimi warnings per visualizzazione più pulita
warnings.filterwarnings('ignore')

def normalize_uint16_to_uint8(volume):
    non_zero = volume[volume > 0]
    if len(non_zero) == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    min_v, max_v = non_zero.min(), non_zero.max()
    norm = np.zeros_like(volume, dtype=np.float32)
    mask = (volume > 0)
    norm[mask] = (volume[mask] - min_v) / (max_v - min_v)
    norm[mask] = (norm[mask] * 255)
    return norm.astype(np.uint8)

def visualizza_tiff_3d_per_crop(image_path, filename):
    """
    Visualizza il file TIFF con viste ortogonali per facilitare la selezione del crop
    Basato su tif_z_range_selector_tifffile.py
    """
    try:
        with Image.open(image_path) as img:
            # Carica tutto il volume se è multi-frame
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f" 📚 Caricamento volume 3D ({img.n_frames} frame)...")

                # Carica tutti i frame in un array 3D
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    frame = np.array(img)
                    frames.append(frame)

                # Crea array 3D: (depth, height, width)
                volume_3d = np.stack(frames, axis=0)
                depth, height, width = volume_3d.shape
                print(f" 📦 Dimensioni volume: {depth}x{height}x{width} (D×H×W)")

                # Calcola slice centrali per ogni piano
                z_center = depth // 2
                y_center = height // 2
                x_center = width // 2

                # Estrai i tre piani ortogonali
                slice_xy = volume_3d[z_center, :, :] # Piano XY (slice lungo Z)
                slice_yz = volume_3d[:, y_center, :] # Piano YZ (slice lungo Y)
                slice_xz = volume_3d[:, :, x_center] # Piano XZ (slice lungo X)

                # Crea figura con 3 subplot
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Piano XY (assiale)
                vmin_xy, vmax_xy = slice_xy.min(), slice_xy.max()
                im1 = axes[0].imshow(slice_xy, cmap='gray', vmin=vmin_xy, vmax=vmax_xy)
                axes[0].set_title(f'Piano XY (Assiale)\nSlice Z={z_center}/{depth-1}\nmin={vmin_xy}, max={vmax_xy}')
                axes[0].set_xlabel('X (Width)')
                axes[0].set_ylabel('Y (Height)')
                if vmax_xy > vmin_xy:
                    plt.colorbar(im1, ax=axes[0], shrink=0.8)

                # Piano YZ (sagittale)
                vmin_yz, vmax_yz = slice_yz.min(), slice_yz.max()
                im2 = axes[1].imshow(slice_yz, cmap='gray', vmin=vmin_yz, vmax=vmax_yz, aspect='auto')
                axes[1].set_title(f'Piano YZ (Sagittale)\nSlice X={x_center}/{width-1}\nmin={vmin_yz}, max={vmax_yz}')
                axes[1].set_xlabel('Z (Depth)')
                axes[1].set_ylabel('Y (Height)')
                if vmax_yz > vmin_yz:
                    plt.colorbar(im2, ax=axes[1], shrink=0.8)

                # Piano XZ (coronale)
                vmin_xz, vmax_xz = slice_xz.min(), slice_xz.max()
                im3 = axes[2].imshow(slice_xz, cmap='gray', vmin=vmin_xz, vmax=vmax_xz, aspect='auto')
                axes[2].set_title(f'Piano XZ (Coronale)\nSlice Y={y_center}/{height-1}\nmin={vmin_xz}, max={vmax_xz}')
                axes[2].set_xlabel('X (Width)')
                axes[2].set_ylabel('Z (Depth)')
                if vmax_xz > vmin_xz:
                    plt.colorbar(im3, ax=axes[2], shrink=0.8)

                # Statistiche del volume completo
                vol_min, vol_max = volume_3d.min(), volume_3d.max()
                vol_mean = volume_3d.mean()
                vol_std = volume_3d.std()

                plt.suptitle(f'Volume 3D: {filename}\n'
                           f'Shape: {depth}×{height}×{width} | '
                           f'Range: [{vol_min}, {vol_max}] | '
                           f'Mean: {vol_mean:.2f} ± {vol_std:.2f}',
                           fontsize=14, fontweight='bold')

                plt.tight_layout()
                plt.show()

                print(f" 📊 Statistiche volume completo:")
                print(f" Range: [{vol_min}, {vol_max}]")
                print(f" Media: {vol_mean:.3f} ± {vol_std:.3f}")
                print(f" Slice centrali: XY={z_center}, YZ={x_center}, XZ={y_center}")
                print(f" Range Z disponibile: 0 - {depth-1}")
                print(f" Range Y disponibile: 0 - {height-1}")
                print(f" Range X disponibile: 0 - {width-1}")

                return volume_3d, depth, height, width
            else:
                print(f" ⚠️ File non è un volume 3D multi-frame")
                return None, 0, 0, 0

    except Exception as e:
        print(f" ❌ Errore nella visualizzazione: {str(e)}")
        return None, 0, 0, 0

def chiedi_range_crop(max_depth, max_height, max_width):
    """
    Chiede all'utente di specificare il range di crop per X, Y, Z
    Basato su tif_z_range_selector_tifffile.py
    """
    print(f"\n🎯 SELEZIONE RANGE CROP")
    print(f" Dimensioni disponibili:")
    print(f"   Z (depth): 0 - {max_depth-1} (totale: {max_depth} slice)")
    print(f"   Y (height): 0 - {max_height-1} (totale: {max_height} pixel)")
    print(f"   X (width): 0 - {max_width-1} (totale: {max_width} pixel)")
    print()
    print(f" 💡 Esempi di input:")
    print(f" - 'tutto' = usa tutto il volume")
    print(f" - '10,50' = dalle slice/pixel 10 alla 50")
    print(f" - '{max_depth//4},{3*max_depth//4}' = range centrale")
    print()

    crop_ranges = {}

    # Range Z
    while True:
        try:
            user_input = input(f" 👉 Range Z [0-{max_depth-1}] (inizio,fine) o 'tutto': ").strip()
            if user_input.lower() in ['tutto', 'all', '']:
                crop_ranges['z'] = (0, max_depth-1)
                print(f" ✅ Z: tutto il volume (0-{max_depth-1})")
                break
            elif ',' in user_input:
                start_str, end_str = user_input.split(',', 1)
                start_z = int(start_str.strip())
                end_z = int(end_str.strip())
                if start_z < 0 or end_z >= max_depth or start_z >= end_z:
                    print(f" ⚠️ Range Z fuori dai limiti! Usa 0-{max_depth-1}")
                    continue
                crop_ranges['z'] = (start_z, end_z)
                print(f" ✅ Z: {start_z}-{end_z} ({end_z-start_z+1} slice)")
                break
            else:
                print(f" ⚠️ Formato non valido per Z! Usa 'inizio,fine' o 'tutto'")
        except ValueError:
            print(f" ⚠️ Inserisci numeri validi per Z!")
        except (KeyboardInterrupt, EOFError):
            return None

    # Range Y  
    while True:
        try:
            user_input = input(f" 👉 Range Y [0-{max_height-1}] (inizio,fine) o 'tutto': ").strip()
            if user_input.lower() in ['tutto', 'all', '']:
                crop_ranges['y'] = (0, max_height-1)
                print(f" ✅ Y: tutto l'asse (0-{max_height-1})")
                break
            elif ',' in user_input:
                start_str, end_str = user_input.split(',', 1)
                start_y = int(start_str.strip())
                end_y = int(end_str.strip())
                if start_y < 0 or end_y >= max_height or start_y >= end_y:
                    print(f" ⚠️ Range Y fuori dai limiti! Usa 0-{max_height-1}")
                    continue
                crop_ranges['y'] = (start_y, end_y)
                print(f" ✅ Y: {start_y}-{end_y} ({end_y-start_y+1} pixel)")
                break
            else:
                print(f" ⚠️ Formato non valido per Y! Usa 'inizio,fine' o 'tutto'")
        except ValueError:
            print(f" ⚠️ Inserisci numeri validi per Y!")
        except (KeyboardInterrupt, EOFError):
            return None

    # Range X
    while True:
        try:
            user_input = input(f" 👉 Range X [0-{max_width-1}] (inizio,fine) o 'tutto': ").strip()
            if user_input.lower() in ['tutto', 'all', '']:
                crop_ranges['x'] = (0, max_width-1)
                print(f" ✅ X: tutto l'asse (0-{max_width-1})")
                break
            elif ',' in user_input:
                start_str, end_str = user_input.split(',', 1)
                start_x = int(start_str.strip())
                end_x = int(end_str.strip())
                if start_x < 0 or end_x >= max_width or start_x >= end_x:
                    print(f" ⚠️ Range X fuori dai limiti! Usa 0-{max_width-1}")
                    continue
                crop_ranges['x'] = (start_x, end_x)
                print(f" ✅ X: {start_x}-{end_x} ({end_x-start_x+1} pixel)")
                break
            else:
                print(f" ⚠️ Formato non valido per X! Usa 'inizio,fine' o 'tutto'")
        except ValueError:
            print(f" ⚠️ Inserisci numeri validi per X!")
        except (KeyboardInterrupt, EOFError):
            return None

    return crop_ranges

def visualizza_volume_croppato(cropped_volume, filename, crop_info):
    """
    Visualizza il volume dopo il crop per conferma
    """
    depth, height, width = cropped_volume.shape
    print(f"\n📦 Volume dopo crop: {depth}x{height}x{width}")

    # Calcola slice centrali
    z_center = depth // 2
    y_center = height // 2
    x_center = width // 2

    # Estrai i tre piani ortogonali
    slice_xy = cropped_volume[z_center, :, :]
    slice_yz = cropped_volume[:, y_center, :]
    slice_xz = cropped_volume[:, :, x_center]

    # Crea figura con 3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Piano XY
    vmin_xy, vmax_xy = slice_xy.min(), slice_xy.max()
    im1 = axes[0].imshow(slice_xy, cmap='gray', vmin=vmin_xy, vmax=vmax_xy)
    axes[0].set_title(f'Croppato XY\nSlice Z={z_center}/{depth-1}')
    axes[0].set_xlabel('X (Width)')
    axes[0].set_ylabel('Y (Height)')
    if vmax_xy > vmin_xy:
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Piano YZ
    vmin_yz, vmax_yz = slice_yz.min(), slice_yz.max()
    im2 = axes[1].imshow(slice_yz, cmap='gray', vmin=vmin_yz, vmax=vmax_yz, aspect='auto')
    axes[1].set_title(f'Croppato YZ\nSlice X={x_center}/{width-1}')
    axes[1].set_xlabel('Z (Depth)')
    axes[1].set_ylabel('Y (Height)')
    if vmax_yz > vmin_yz:
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Piano XZ
    vmin_xz, vmax_xz = slice_xz.min(), slice_xz.max()
    im3 = axes[2].imshow(slice_xz, cmap='gray', vmin=vmin_xz, vmax=vmax_xz, aspect='auto')
    axes[2].set_title(f'Croppato XZ\nSlice Y={y_center}/{height-1}')
    axes[2].set_xlabel('X (Width)')
    axes[2].set_ylabel('Z (Depth)')
    if vmax_xz > vmin_xz:
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.suptitle(f'Volume Croppato: {filename}\n'
               f'Z: {crop_info["z"][0]}-{crop_info["z"][1]} | '
               f'Y: {crop_info["y"][0]}-{crop_info["y"][1]} | '
               f'X: {crop_info["x"][0]}-{crop_info["x"][1]}',
               fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

def chiedi_destinazione_folder():
    """
    Chiede all'utente dove salvare l'immagine processata
    Basato su check-black-tiff-immediate.py
    """
    print(f"\n📁 SELEZIONE CARTELLA DESTINAZIONE")
    print(f" Dove vuoi salvare l'immagine processata?")
    print()
    print(f" 📋 OPZIONI DISPONIBILI:")
    print(f" [B] - Cartella 'black' (immagini scure/nere)")
    print(f" [C] - Cartella 'check' (da controllare)")
    print(f" [N] - Cartella principale (normale)")
    print()

    while True:
        try:
            scelta = input(" 👉 Scelta [B/C/N]: ").strip().upper()
            if scelta in ['B', 'BLACK', 'NERO']:
                print(" 🖤 Salverà in cartella 'black'")
                return 'black'
            elif scelta in ['C', 'CHECK']:
                print(" 🔍 Salverà in cartella 'check'")
                return 'check'
            elif scelta in ['N', 'NORMAL', 'NORMALE', '']:
                print(" ✅ Salverà in cartella principale")
                return 'normal'
            else:
                print(" ⚠️ Scelta non valida. Usa: B (black), C (check), N (normale)")
        except (KeyboardInterrupt, EOFError):
            print("\n 🛑 Interruzione - uscita")
            return None

def setup_output_folders(base_output_path):
    """
    Crea le cartelle di output necessarie
    """
    folders = {
        'main': base_output_path,
        'black': os.path.join(base_output_path, 'black'),
        'check': os.path.join(base_output_path, 'check'),
        'analysis_plots': os.path.join(base_output_path, 'analysis_plots')
    }

    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
        if folder_name != 'main':
            print(f"📁 Cartella '{folder_name}' pronta: {folder_path}")

    return folders

def analyze_and_resize_tiff_volumes_interactive(input_folder, output_folder, target_shape=(512, 512), preserve_structure=True):
    """
    VERSIONE INTERATTIVA MODIFICATA del processamento TIFF
    Include visualizzazione, crop interattivo e selezione cartella destinazione
    """
    # Crea cartelle se non esistono
    folders = setup_output_folders(output_folder)

    # Trova tutti i file .tif ricorsivamente nelle sottocartelle
    input_path = Path(input_folder)
    tif_files = list(input_path.glob("**/*.tif")) # Ricerca ricorsiva
    tif_files_str = [str(f) for f in tif_files] # Converti in stringhe per compatibilità

    if not tif_files:
        print(f"Nessun file .tif trovato nella cartella {input_folder} e sue sottocartelle")
        return

    print(f"Trovati {len(tif_files)} file .tif da processare (incluse sottocartelle)...")

    # Mostra struttura delle cartelle trovate
    subdirs_found = set()
    for tif_file in tif_files:
        relative_path = tif_file.relative_to(input_path)
        if len(relative_path.parts) > 1: # Ha sottocartelle
            subdir = relative_path.parent
            subdirs_found.add(str(subdir))

    if subdirs_found:
        print(f"Sottocartelle trovate: {sorted(subdirs_found)}")

    # Liste per raccogliere statistiche
    volume_stats = []
    processing_errors = []

    print("\n" + "="*70)
    print("🎯 MODALITÀ INTERATTIVA CON CROP E SELEZIONE CARTELLA")
    print(" • Ogni file viene visualizzato con viste ortogonali")
    print(" • Puoi specificare il crop su X, Y, Z")
    print(" • Scegli la cartella di destinazione (black/check/normal)")
    print(" • Il file viene ridimensionato a 512x512 e salvato")
    print("="*70)

    # Processa ogni file interattivamente
    for i, tif_path in enumerate(tif_files_str):
        try:
            # Calcola percorso relativo per mantenere struttura
            tif_path_obj = Path(tif_path)
            relative_path = tif_path_obj.relative_to(input_path)

            print(f"\n[{i+1:3d}/{len(tif_files)}] 📄 Processando: {tif_path_obj.name}")
            print("-" * 50)

            # ========== NUOVA VISUALIZZAZIONE INTERATTIVA ==========
            print("🖼️ Visualizzazione volume 3D per selezione crop...")
            volume_3d, depth, height, width = visualizza_tiff_3d_per_crop(tif_path, tif_path_obj.name)

            if volume_3d is None:
                print(f"SKIP: {relative_path} - Non è un volume 3D valido")
                continue

            # Normalizza il volume
            volume_3d = normalize_uint16_to_uint8(volume_3d)

            # ========== RICHIESTA CROP INTERATTIVA ==========
            crop_ranges = chiedi_range_crop(depth, height, width)
            if crop_ranges is None:
                print("⏭️ File saltato dall'utente")
                continue

            # Applica il crop
            z_start, z_end = crop_ranges['z']
            y_start, y_end = crop_ranges['y'] 
            x_start, x_end = crop_ranges['x']

            volume = volume_3d[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]

            print(f"\n🔪 Crop applicato:")
            print(f" Z: {z_start}-{z_end} | Y: {y_start}-{y_end} | X: {x_start}-{x_end}")
            print(f" Nuove dimensioni: {volume.shape}")

            # ========== VISUALIZZA VOLUME CROPPATO ==========
            print("\n🖼️ Visualizzazione volume dopo crop...")
            visualizza_volume_croppato(volume, tif_path_obj.name, crop_ranges)

            # ========== SELEZIONE CARTELLA DESTINAZIONE ==========
            folder_choice = chiedi_destinazione_folder()
            if folder_choice is None:
                print("⏭️ File saltato dall'utente")
                continue

            # Determina cartella di output
            if folder_choice == 'black':
                output_base = folders['black']
            elif folder_choice == 'check':
                output_base = folders['check'] 
            else:  # normal
                output_base = folders['main']

            if len(volume.shape) < 3:
                print(f"SKIP: {relative_path} - Non è un volume 3D dopo crop")
                continue

            # Raccogli statistiche (incluso percorso relativo)
            file_stats = {
                'filename': tif_path_obj.name,
                'relative_path': str(relative_path),
                'subfolder': str(relative_path.parent) if len(relative_path.parts) > 1 else 'root',
                'original_shape': volume.shape,
                'num_slices': volume.shape[0],
                'original_height': volume.shape[1],
                'original_width': volume.shape[2],
                'dtype': str(volume.dtype),
                'min_value': volume.min(),
                'max_value': volume.max(),
                'mean_value': volume.mean(),
                'std_value': volume.std(),
                'file_size_mb': os.path.getsize(tif_path) / (1024**2),
                'crop_applied': f"Z:{z_start}-{z_end}_Y:{y_start}-{y_end}_X:{x_start}-{x_end}",
                'destination_folder': folder_choice
            }

            # Ridimensiona volume a 512x512
            print(f"\n🔧 Ridimensionamento a {target_shape}...")
            resized_volume = resize(
                volume,
                (volume.shape[0], target_shape[0], target_shape[1]),
                order=1,
                preserve_range=True,
                anti_aliasing=True
            ).astype(volume.dtype)

            # Determina percorso di output
            if preserve_structure:
                output_path = Path(output_base) / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True) # Crea sottocartelle se necessario
            else:
                # Salva tutto nella cartella principale, rinomina se conflitti
                output_filename = f"{relative_path.parent}_{tif_path_obj.name}".replace(os.sep, "_")
                if str(relative_path.parent) == ".": # File nella cartella root
                    output_filename = tif_path_obj.name
                output_path = Path(output_base) / output_filename

            # Salva volume ridimensionato
            tiff.imwrite(str(output_path), resized_volume)

            # Aggiorna statistiche post-processing
            file_stats['output_path'] = str(output_path.relative_to(Path(output_folder)))
            file_stats['resized_shape'] = resized_volume.shape
            file_stats['resized_file_size_mb'] = os.path.getsize(str(output_path)) / (1024**2)
            file_stats['size_reduction_percent'] = (1 - file_stats['resized_file_size_mb'] / file_stats['file_size_mb']) * 100

            volume_stats.append(file_stats)

            print(f" ✅ File salvato: {output_path}")
            print(f" 📏 Dimensioni finali: {resized_volume.shape}")
            print(f" 📁 Cartella: {folder_choice}")

        except Exception as e:
            # Gestione errori con percorso relativo
            error_info = {
                'filename': Path(tif_path).name,
                'relative_path': str(Path(tif_path).relative_to(input_path)),
                'error': str(e)
            }
            processing_errors.append(error_info)
            print(f"ERRORE processando {Path(tif_path).relative_to(input_path)}: {e}")
            continue

    # Converti in DataFrame per analisi
    df_stats = pd.DataFrame(volume_stats)
    if df_stats.empty:
        print("Nessun file processato con successo!")
        return

    # Genera grafici e statistiche
    _generate_analysis_plots(df_stats, folders['analysis_plots'])
    _print_summary_statistics_interactive(df_stats, processing_errors, input_folder, output_folder)

    # Salva statistiche in CSV
    stats_csv_path = os.path.join(output_folder, "volume_statistics_interactive.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"Statistiche salvate in: {stats_csv_path}")
    print("\nProcessing completato!")

def _generate_analysis_plots(df_stats, plots_folder):
    """Genera e salva grafici di analisi delle statistiche dei volumi"""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    # 1. Distribuzione del numero di slice
    plt.figure(figsize=(10, 6))
    plt.hist(df_stats['num_slices'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Numero di Slice')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione del Numero di Slice per Volume')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_folder, 'slice_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Distribuzione per cartella destinazione
    if 'destination_folder' in df_stats.columns:
        plt.figure(figsize=(10, 6))
        dest_counts = df_stats['destination_folder'].value_counts()
        colors = {'black': 'black', 'check': 'orange', 'normal': 'green'}
        plt.pie(dest_counts.values, labels=dest_counts.index, autopct='%1.1f%%', 
               colors=[colors.get(x, 'gray') for x in dest_counts.index])
        plt.title('Distribuzione File per Cartella Destinazione')
        plt.savefig(os.path.join(plots_folder, 'destination_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Riduzione dimensione file
    plt.figure(figsize=(10, 6))
    plt.hist(df_stats['size_reduction_percent'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Riduzione Dimensione File (%)')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione della Riduzione Dimensione File')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_folder, 'size_reduction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Grafici salvati in: {plots_folder}")

def _print_summary_statistics_interactive(df_stats, processing_errors, input_folder, output_folder):
    """Stampa statistiche riassuntive del processing interattivo"""
    print("\n" + "="*60)
    print("STATISTICHE RIASSUNTIVE DEL PROCESSING INTERATTIVO")
    print("="*60)
    print(f"File processati con successo: {len(df_stats)}")
    print(f"File con errori: {len(processing_errors)}")

    # Statistiche per cartelle destinazione
    if 'destination_folder' in df_stats.columns:
        dest_counts = df_stats['destination_folder'].value_counts()
        print(f"\nDISTRIBUZIONE PER CARTELLA:")
        for folder, count in dest_counts.items():
            print(f" {folder}: {count} file")

    print(f"\nNUMERO DI SLICE:")
    print(f" Min: {df_stats['num_slices'].min()}")
    print(f" Max: {df_stats['num_slices'].max()}")
    print(f" Media: {df_stats['num_slices'].mean():.1f}")

    if processing_errors:
        print(f"\nERRORI DI PROCESSING:")
        for error in processing_errors:
            print(f" {error['relative_path']}: {error['error']}")

if __name__ == "__main__":
    # Esempio di utilizzo
    #input_folder = 'F:\\Organoids\\Noyaux\\Cystiques\\Nice'
    #output_folder = 'F:\\Organoids\\Cystiques_Nice_Interactive_Crop'
    input_folder = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data"
    output_folder = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data/Processed_Interactive_Crop"

    # Esegui la versione interattiva
    analyze_and_resize_tiff_volumes_interactive(input_folder, output_folder, target_shape=(512, 512), preserve_structure=False)

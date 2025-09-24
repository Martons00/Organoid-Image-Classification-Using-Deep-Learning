
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
from matplotlib.ticker import MultipleLocator

def add_grid(ax,step_x,step_y,color='r',a=0.35,lw=0.6) :
    ax.xaxis.set_major_locator(MultipleLocator(step_x))
    ax.yaxis.set_major_locator(MultipleLocator(step_y))
    ax.grid(True,which='major',color=color,alpha=a,linewidth=lw)
    ax.tick_params(axis='x',rotation=90)

# =================== FUNZIONI CROP AUTOMATICO ===================

def automatic_crop_detection(volume_3d, method='otsu', padding_ratio=0.05, min_size_ratio=0.1, show_debug=False):
    """
    Rileva automaticamente i confini dell'organoide per il crop usando vari metodi di thresholding

    Args:
        volume_3d: Volume 3D (numpy array)
        method: 'otsu', 'mean', 'median', 'percentile', 'adaptive'
        padding_ratio: Percentuale di padding da aggiungere ai bordi (default 5%)
        min_size_ratio: Dimensione minima dell'oggetto rispetto al volume totale
        show_debug: Mostra immagini di debug

    Returns:
        dict: {'x': (x_start, x_end), 'y': (y_start, y_end), 'z': (z_start, z_end)}
    """
    from skimage import filters, measure, morphology
    import numpy as np
    import matplotlib.pyplot as plt

    print(f"🔍 Rilevamento automatico crop con metodo: {method}")
    print(f"📦 Dimensioni volume originale: {volume_3d.shape}")

    # 1. PREPROCESSING - Normalizza e pulisci il volume
    volume_normalized = volume_3d.astype(np.float32)

    # Rimuovi outlier estremi (0.1% e 99.9% percentili)
    non_zero_mask = volume_normalized > 0
    if np.sum(non_zero_mask) == 0:
        print("⚠️ Volume completamente vuoto!")
        return None

    p1, p99 = np.percentile(volume_normalized[non_zero_mask], [0.1, 99.9])
    volume_normalized = np.clip(volume_normalized, p1, p99)

    # Normalizza 0-1
    if volume_normalized.max() > volume_normalized.min():
        volume_normalized = (volume_normalized - volume_normalized.min()) / (volume_normalized.max() - volume_normalized.min())

    # 2. SELEZIONE THRESHOLD DINAMICO
    non_zero_pixels = volume_normalized[volume_normalized > 0]

    if method == 'otsu':
        # Otsu threshold su pixel non-zero
        threshold = filters.threshold_otsu(non_zero_pixels)
        print(f"📊 Otsu threshold: {threshold:.4f}")

    elif method == 'mean':
        # Threshold basato su media + std
        mean_val = np.mean(non_zero_pixels)
        std_val = np.std(non_zero_pixels)
        threshold = mean_val + 0.5 * std_val  # Soglia conservativa
        print(f"📊 Mean threshold: {threshold:.4f} (mean: {mean_val:.4f}, std: {std_val:.4f})")

    elif method == 'median':
        # Threshold basato su mediana
        threshold = np.median(non_zero_pixels)
        print(f"📊 Median threshold: {threshold:.4f}")

    elif method == 'percentile':
        # Threshold al 75° percentile (distingue background da oggetto)
        threshold = np.percentile(non_zero_pixels, 75)
        print(f"📊 75th percentile threshold: {threshold:.4f}")

    elif method == 'adaptive':
        # Threshold adattivo basato su istogramma
        hist, bins = np.histogram(non_zero_pixels, bins=256)
        # Trova il secondo picco nell'istogramma (primo = background, secondo = oggetto)
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.05:
                peaks.append((bins[i], hist[i]))

        if len(peaks) >= 2:
            # Usa il valore tra i due picchi principali
            peaks.sort(key=lambda x: x[1], reverse=True)  # Ordina per intensità
            peak1, peak2 = peaks[0][0], peaks[1][0]
            threshold = (peak1 + peak2) / 2
        else:
            # Fallback su percentile se non trova due picchi
            threshold = np.percentile(non_zero_pixels, 70)

        print(f"📊 Adaptive threshold: {threshold:.4f} (peaks found: {len(peaks)})")
    else:
        # Default fallback
        threshold = np.percentile(non_zero_pixels, 75)
        print(f"📊 Default threshold: {threshold:.4f}")

    # 3. CREAZIONE MASCHERA BINARIA
    binary_mask = volume_normalized > threshold

    # 4. PULIZIA MORFOLOGICA
    # Rimuovi piccoli oggetti rumorosi
    min_size = int(np.prod(volume_3d.shape) * min_size_ratio)
    try:
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
        # Chiusura morfologica per riempire buchi
        binary_mask = morphology.binary_closing(binary_mask, morphology.ball(3))
    except:
        # Fallback senza operazioni morfologiche se falliscono
        print("⚠️ Operazioni morfologiche fallite, uso maschera semplice")

    # 5. TROVA BOUNDING BOX dell'oggetto principale
    if np.sum(binary_mask) == 0:
        print("⚠️ Nessun oggetto rilevato con questa soglia!")
        return None

    # Trova coordinate non-zero
    coords = np.where(binary_mask)
    z_coords, y_coords, x_coords = coords

    # Calcola bounding box
    z_min, z_max = z_coords.min(), z_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # 6. AGGIUNGI PADDING
    depth, height, width = volume_3d.shape

    z_padding = int((z_max - z_min) * padding_ratio)
    y_padding = int((y_max - y_min) * padding_ratio)
    x_padding = int((x_max - x_min) * padding_ratio)

    z_start = max(0, z_min - z_padding)
    z_end = min(depth - 1, z_max + z_padding)
    y_start = max(0, y_min - y_padding)
    y_end = min(height - 1, y_max + y_padding)
    x_start = max(0, x_min - x_padding)
    x_end = min(width - 1, x_max + x_padding)

    crop_ranges = {
        'z': (z_start, z_end),
        'y': (y_start, y_end),
        'x': (x_start, x_end)
    }

    # 7. STATISTICHE E DEBUG
    original_volume = depth * height * width
    cropped_volume = (z_end - z_start + 1) * (y_end - y_start + 1) * (x_end - x_start + 1)
    volume_reduction = (1 - cropped_volume / original_volume) * 100

    print(f"📏 Bounding box rilevata:")
    print(f"   Z: {z_start} - {z_end} ({z_end - z_start + 1} slice)")
    print(f"   Y: {y_start} - {y_end} ({y_end - y_start + 1} pixel)")
    print(f"   X: {x_start} - {x_end} ({x_end - x_start + 1} pixel)")
    print(f"📉 Riduzione volume: {volume_reduction:.1f}% (era {original_volume:,} → ora {cropped_volume:,} voxel)")

    return crop_ranges

def automatic_crop_with_fallback(volume_3d, filename, interactive=True, auto_method='otsu'):
    """
    Versione ibrida: prova crop automatico, con fallback manuale se necessario
    """
    print(f"\n🤖 TENTATIVO CROP AUTOMATICO: {auto_method}")

    # Prova crop automatico
    crop_ranges = automatic_crop_detection(volume_3d, method=auto_method, show_debug=False)

    if crop_ranges is None:
        print("❌ Crop automatico fallito - passaggio a modalità manuale")
        if interactive:
            return chiedi_range_crop(volume_3d.shape[0], volume_3d.shape[1], volume_3d.shape[2])
        else:
            return None

    if interactive:
        # Mostra risultato e chiedi conferma
        print("\n✅ Crop automatico completato!")

        # Applica crop temporaneo per visualizzazione
        z_start, z_end = crop_ranges['z']
        y_start, y_end = crop_ranges['y']
        x_start, x_end = crop_ranges['x']

        temp_cropped = volume_3d[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
        visualizza_volume_croppato(temp_cropped, filename, crop_ranges)

        print("\n🤔 Il crop automatico ti sembra corretto?")
        print("A - Accetta crop automatico")
        print("M - Modalità manuale (scegli tu i range)")
        print("R - Riprova con metodo diverso")
        print("S - Salta questo file")

        while True:
            try:
                choice = input("Scelta [A/M/R/S]: ").strip().upper()
                if choice == 'A':
                    print("✅ Crop automatico accettato!")
                    return crop_ranges
                elif choice == 'M':
                    print("👤 Passaggio a modalità manuale...")
                    return chiedi_range_crop(volume_3d.shape[0], volume_3d.shape[1], volume_3d.shape[2])
                elif choice == 'R':
                    print("🔄 Riprova con metodo diverso...")
                    methods = ['otsu', 'mean', 'percentile', 'adaptive']
                    available_methods = [m for m in methods if m != auto_method]

                    print(f"Metodi disponibili: {', '.join(available_methods)}")
                    new_method = input(f"Nuovo metodo [{'/'.join(available_methods)}]: ").strip().lower()

                    if new_method in available_methods:
                        return automatic_crop_with_fallback(volume_3d, filename, interactive, new_method)
                    else:
                        print("Metodo non valido, uso 'mean'")
                        return automatic_crop_with_fallback(volume_3d, filename, interactive, 'mean')
                elif choice == 'S':
                    print("⏭️ File saltato")
                    return None
                else:
                    print("Scelta non valida. Usa A, M, R, o S")
            except (KeyboardInterrupt, EOFError):
                return None
    else:
        # Modalità non interattiva - accetta automaticamente
        return crop_ranges
    
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

# =================== SISTEMA TRACKING FILE PROCESSATI ===================

def load_processed_files(log_path):
    """
    Carica l'elenco dei file già processati dal file log
    Returns: set di percorsi relativi già processati
    """
    processed_files = set()
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Estrai solo il percorso relativo dalla riga del log
                        if ' | ' in line:
                            relative_path = line.split(' | ')[0].strip()
                            processed_files.add(relative_path)
                        else:
                            processed_files.add(line)
            print(f"📋 Caricati {len(processed_files)} file già processati dal log")
        except Exception as e:
            print(f"⚠️ Errore caricamento log file: {e}")
            processed_files = set()
    else:
        print(f"📋 File log non trovato, iniziando nuovo processing")

    return processed_files

def save_processed_file(log_path, relative_path, status, details=""):
    """
    Salva un file come processato nel file log
    Args:
        log_path: percorso del file log
        relative_path: percorso relativo del file
        status: SUCCESS, ERROR, SKIP
        details: dettagli aggiuntivi (cartella destinazione, errore, ecc.)
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{relative_path} | {status} | {timestamp}"
        if details:
            log_entry += f" | {details}"
        log_entry += "\n"

        # Crea cartella se non esiste
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Appende al file log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"⚠️ Errore salvataggio log: {e}")

def initialize_log_file(log_path):
    """
    Inizializza il file log con header se è un nuovo file
    """
    if not os.path.exists(log_path):
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("# LOG FILE PROCESSING INTERATTIVO\n")
                f.write("# Formato: relative_path | status | timestamp | details\n")
                f.write(f"# Creato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
            print(f"📝 Creato nuovo file log: {log_path}")
        except Exception as e:
            print(f"⚠️ Errore creazione file log: {e}")

def print_processing_status(total_files, processed_count, skipped_count):
    """
    Stampa lo stato del processing
    """
    remaining = total_files - processed_count - skipped_count
    progress_percent = ((processed_count + skipped_count) / total_files) * 100 if total_files > 0 else 0

    print(f"\n📊 STATO PROCESSING:")
    print(f" Totale file: {total_files}")
    print(f" Già processati (skippati): {skipped_count}")
    print(f" Processati in questa sessione: {processed_count}")
    print(f" Rimanenti: {remaining}")
    print(f" Progresso: {progress_percent:.1f}%")

# =================== FINE SISTEMA TRACKING ===================

def visualizza_tiff_3d_per_crop(image_path, filename):
    """
    Visualizza il file TIFF con viste ortogonali per facilitare la selezione del crop
    Basato su tif_z_range_selector_tifffile.py
    """
    try:
        volume_3d = io.imread(image_path)
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
        add_grid(axes[0],step_x=100,step_y=100)
        add_grid(axes[0],step_x=200,step_y=200,color='g')
        if vmax_xy > vmin_xy:
            plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # Piano YZ (sagittale)
        vmin_yz, vmax_yz = slice_yz.min(), slice_yz.max()
        im2 = axes[1].imshow(slice_yz, cmap='gray', vmin=vmin_yz, vmax=vmax_yz, aspect='auto')
        axes[1].set_title(f'Piano YZ (Sagittale)\nSlice X={x_center}/{width-1}\nmin={vmin_yz}, max={vmax_yz}')
        axes[1].set_ylabel('Z (Depth)')
        axes[1].set_xlabel('Y (Height)')
        add_grid(axes[1],step_x=100,step_y=10)
        add_grid(axes[1],step_x=200,step_y=20,color='g')
        if vmax_yz > vmin_yz:
            plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # Piano XZ (coronale)
        vmin_xz, vmax_xz = slice_xz.min(), slice_xz.max()
        im3 = axes[2].imshow(slice_xz, cmap='gray', vmin=vmin_xz, vmax=vmax_xz, aspect='auto')
        axes[2].set_title(f'Piano XZ (Coronale)\nSlice Y={y_center}/{height-1}\nmin={vmin_xz}, max={vmax_xz}')
        axes[2].set_xlabel('X (Width)')
        axes[2].set_ylabel('Z (Depth)')
        add_grid(axes[2],step_x=100,step_y=10)
        add_grid(axes[2],step_x=200,step_y=20,color='g')
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
        plt.show(block=True)  # Forza visualizzazione

        print(f" 📊 Statistiche volume completo:")
        print(f" Range: [{vol_min}, {vol_max}]")
        print(f" Media: {vol_mean:.3f} ± {vol_std:.3f}")
        print(f" Slice centrali: XY={z_center}, YZ={x_center}, XZ={y_center}")
        print(f" Range Z disponibile: 0 - {depth-1}")
        print(f" Range Y disponibile: 0 - {height-1}")
        print(f" Range X disponibile: 0 - {width-1}")

            # Range Z
        while True:
            try:
                user_input = input(f"Proseguo o non ne vale la pena? : ").strip()
                if user_input.lower() in ['s', 'y', '']:
                    break
                elif user_input.lower() == 'n':
                    return None, 0, 0, 0
                else:
                    print(f" ⚠️ Formato non valido")
            except ValueError:
                print(f" ⚠️ Inserisci numeri validi")
            except (KeyboardInterrupt, EOFError):
                return None, 0, 0, 0

        return volume_3d, depth, height, width

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
    plt.show(block=True)  # Forza visualizzazione

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
        'analysis_plots': os.path.join(base_output_path, 'analysis_plots'),
        'logs': os.path.join(base_output_path, 'logs')  # AGGIUNTA cartella logs
    }

    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
        if folder_name not in ['main']:
            print(f"📁 Cartella '{folder_name}' pronta: {folder_path}")

    return folders

def analyze_and_resize_tiff_volumes_interactive(input_folder, output_folder, target_shape=(512, 512), preserve_structure=True):
    """
    VERSIONE INTERATTIVA MODIFICATA del processamento TIFF
    Include visualizzazione, crop interattivo, selezione cartella destinazione E TRACKING FILE PROCESSATI
    """
    # Crea cartelle se non esistono
    folders = setup_output_folders(output_folder)

    # ============== INIZIALIZZAZIONE SISTEMA TRACKING ==============
    log_file_path = os.path.join(folders['logs'], 'processed_files.txt')
    initialize_log_file(log_file_path)
    processed_files_set = load_processed_files(log_file_path)
    # ============== FINE INIZIALIZZAZIONE TRACKING ==============

    # Trova tutti i file .tif ricorsivamente nelle sottocartelle
    input_path = Path(input_folder)
    tif_files = list(input_path.glob("**/*.tif")) # Ricerca ricorsiva
    tif_files_str = [str(f) for f in tif_files] # Converti in stringhe per compatibilità

    if not tif_files:
        print(f"Nessun file .tif trovato nella cartella {input_folder} e sue sottocartelle")
        return

    print(f"Trovati {len(tif_files)} file .tif da processare (incluse sottocartelle)...")

    # ============== FILTRO FILE GIA' PROCESSATI ==============
    files_to_process = []
    skipped_files = []

    for tif_file in tif_files:
        relative_path = str(tif_file.relative_to(input_path))
        if relative_path in processed_files_set:
            skipped_files.append(relative_path)
        else:
            files_to_process.append(str(tif_file))

    print(f"\n📋 ANALISI FILE:")
    print(f" File totali trovati: {len(tif_files)}")
    print(f" File già processati (da saltare): {len(skipped_files)}")
    print(f" File da processare: {len(files_to_process)}")

    if len(skipped_files) > 0:
        print(f"\n📝 FILE GIÀ PROCESSATI (SALTATI):")
        for i, skipped in enumerate(skipped_files[:10]):  # Mostra solo i primi 10
            print(f" {i+1:2d}. {skipped}")
        if len(skipped_files) > 10:
            print(f" ... e altri {len(skipped_files)-10} file")

    if len(files_to_process) == 0:
        print("\n🎉 Tutti i file sono già stati processati!")
        return

    # Chiedi conferma se ci sono file già processati
    if len(skipped_files) > 0:
        print(f"\n❓ VUOI CONTINUARE?")
        print(f" Verranno processati solo {len(files_to_process)} file nuovi.")
        print(f" I {len(skipped_files)} file già processati verranno saltati.")

        while True:
            try:
                choice = input(" 👉 Continuare? [S/N]: ").strip().upper()
                if choice in ['S', 'SI', 'Y', 'YES', '']:
                    break
                elif choice in ['N', 'NO']:
                    print(" 🛑 Operazione annullata dall'utente")
                    return
                else:
                    print(" ⚠️ Risposta non valida. Usa S (si) o N (no)")
            except (KeyboardInterrupt, EOFError):
                print("\n 🛑 Interruzione - uscita")
                return
    # ============== FINE FILTRO FILE GIA' PROCESSATI ==============

    # Mostra struttura delle cartelle trovate
    subdirs_found = set()
    for tif_file_path in files_to_process:
        tif_file = Path(tif_file_path)
        relative_path = tif_file.relative_to(input_path)
        if len(relative_path.parts) > 1: # Ha sottocartelle
            subdir = relative_path.parent
            subdirs_found.add(str(subdir))

    if subdirs_found:
        print(f"Sottocartelle da processare: {sorted(subdirs_found)}")

    # Liste per raccogliere statistiche
    volume_stats = []
    processing_errors = []
    files_processed_this_session = 0

    print("\n" + "="*70)
    print("🎯 MODALITÀ INTERATTIVA CON CROP E SELEZIONE CARTELLA + TRACKING")
    print(" • Ogni file viene visualizzato con viste ortogonali")
    print(" • Puoi specificare il crop su X, Y, Z")
    print(" • Scegli la cartella di destinazione (black/check/normal)")
    print(" • Il file viene ridimensionato a 512x512 e salvato")
    print(" • Progressi salvati automaticamente - puoi interrompere e riprendere!")
    print("="*70)

    # Processa ogni file interattivamente
    for i, tif_path in enumerate(files_to_process):
        try:
            # Calcola percorso relativo per mantenere struttura
            tif_path_obj = Path(tif_path)
            relative_path = str(tif_path_obj.relative_to(input_path))

            print(f"\n[{i+1:3d}/{len(files_to_process)}] 📄 Processando: {tif_path_obj.name}")
            print("-" * 50)

            # ============== STATUS UPDATE ==============
            print_processing_status(len(tif_files), files_processed_this_session, len(skipped_files))

            # ========== NUOVA VISUALIZZAZIONE INTERATTIVA ==========
            print("\n🖼️ Visualizzazione volume 3D per selezione crop...")
            volume_3d, depth, height, width = visualizza_tiff_3d_per_crop(tif_path, tif_path_obj.name)

            if volume_3d is None:
                print(f"SKIP: {relative_path} - Non è un volume 3D valido")
                save_processed_file(log_file_path, relative_path, "SKIP", "Non è un volume 3D valido")
                continue


            # ========== RICHIESTA CROP INTERATTIVA ==========
            crop_ranges = chiedi_range_crop(depth, height, width)
            #crop_ranges = automatic_crop_detection(volume_3d)

            if crop_ranges is None:
                print("⏭️ File saltato dall'utente")
                save_processed_file(log_file_path, relative_path, "SKIP", "Saltato dall'utente")
                continue

            # Applica il crop
            z_start, z_end = crop_ranges['z']
            y_start, y_end = crop_ranges['y'] 
            x_start, x_end = crop_ranges['x']

            
            dif_x = x_end - x_start
            dif_y = y_end - y_start

            if dif_x > dif_y :
                y_end = y_start + dif_x
                print("Correggo la y per rendere il tutto quadrato.")
                if y_end > width : 
                    y_end = width
                    y_start = y_end - dif_x
            else : 
                x_end = x_start + dif_y 
                print("Correggo la x per rendere il tutto quadrato.")
                if x_end > height :
                    x_end = height
                    x_start = x_end - dif_y

            volume = volume_3d[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
            print(" Normalizzo in uint8.")
            volume = normalize_uint16_to_uint8(volume)

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
                save_processed_file(log_file_path, relative_path, "SKIP", "Saltato dall'utente - selezione cartella")
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
                save_processed_file(log_file_path, relative_path, "SKIP", "Non è un volume 3D dopo crop")
                continue

            # Raccogli statistiche (incluso percorso relativo)
            file_stats = {
                'filename': tif_path_obj.name,
                'relative_path': relative_path,
                'subfolder': str(tif_path_obj.relative_to(input_path).parent) if len(tif_path_obj.relative_to(input_path).parts) > 1 else 'root',
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
                output_path = Path(output_base) / tif_path_obj.relative_to(input_path)
                output_path.parent.mkdir(parents=True, exist_ok=True) # Crea sottocartelle se necessario
            else:
                # Salva tutto nella cartella principale, rinomina se conflitti
                relative_path_obj = tif_path_obj.relative_to(input_path)
                output_filename = f"{relative_path_obj.parent}_{tif_path_obj.name}".replace(os.sep, "_")
                if str(relative_path_obj.parent) == ".": # File nella cartella root
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
            files_processed_this_session += 1

            print(f" ✅ File salvato: {output_path}")
            print(f" 📏 Dimensioni finali: {resized_volume.shape}")
            print(f" 📁 Cartella: {folder_choice}")

            # ============== SALVA NEL LOG FILE ==============
            crop_details = f"Crop Z:{z_start}-{z_end}_Y:{y_start}-{y_end}_X:{x_start}-{x_end} | Dest: {folder_choice}"
            save_processed_file(log_file_path, relative_path, "SUCCESS", crop_details)

        except KeyboardInterrupt:
            print("\n\n🛑 INTERRUZIONE RILEVATA!")
            print("💾 Progresso salvato automaticamente nel file log.")
            print("🔄 Puoi riprendere l'esecuzione dello script e continuerà da dove hai interrotto.")
            break
        except Exception as e:
            # Gestione errori con percorso relativo
            error_info = {
                'filename': Path(tif_path).name,
                'relative_path': str(Path(tif_path).relative_to(input_path)),
                'error': str(e)
            }
            processing_errors.append(error_info)
            print(f"ERRORE processando {Path(tif_path).relative_to(input_path)}: {e}")

            # ============== SALVA ERRORE NEL LOG FILE ==============
            save_processed_file(log_file_path, str(Path(tif_path).relative_to(input_path)), "ERROR", str(e))
            continue

    # Converti in DataFrame per analisi
    df_stats = pd.DataFrame(volume_stats)
    if df_stats.empty:
        print("\n📊 Nessun file processato con successo in questa sessione!")
        if files_processed_this_session == 0 and len(skipped_files) > 0:
            print("(Tutti i file erano già stati processati precedentemente)")
        return

    # Genera grafici e statistiche
    _generate_analysis_plots(df_stats, folders['analysis_plots'])
    _print_summary_statistics_interactive(df_stats, processing_errors, input_folder, output_folder, len(skipped_files))

    # Salva statistiche in CSV
    stats_csv_path = os.path.join(output_folder, f"volume_statistics_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"\n💾 Statistiche salvate in: {stats_csv_path}")
    print(f"📝 Log dei file processati salvato in: {log_file_path}")
    print("\n🎉 Processing completato!")

def _generate_analysis_plots(df_stats, plots_folder):
    """Genera e salva grafici di analisi delle statistiche dei volumi"""
    # Usa backend Agg solo per salvare i grafici di analisi finale
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')

    plt.style.use('default')

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

    # Ripristina backend originale
    plt.switch_backend(original_backend)

    print(f"📊 Grafici salvati in: {plots_folder}")

def _print_summary_statistics_interactive(df_stats, processing_errors, input_folder, output_folder, skipped_count=0):
    """Stampa statistiche riassuntive del processing interattivo"""
    print("\n" + "="*60)
    print("📊 STATISTICHE RIASSUNTIVE DEL PROCESSING INTERATTIVO")
    print("="*60)
    print(f"File processati in questa sessione: {len(df_stats)}")
    print(f"File saltati (già processati): {skipped_count}")
    print(f"File con errori: {len(processing_errors)}")

    # Statistiche per cartelle destinazione
    if 'destination_folder' in df_stats.columns and len(df_stats) > 0:
        dest_counts = df_stats['destination_folder'].value_counts()
        print(f"\nDISTRIBUZIONE PER CARTELLA (SESSIONE CORRENTE):")
        for folder, count in dest_counts.items():
            print(f" {folder}: {count} file")

    if len(df_stats) > 0:
        print(f"\nNUMERO DI SLICE:")
        print(f" Min: {df_stats['num_slices'].min()}")
        print(f" Max: {df_stats['num_slices'].max()}")
        print(f" Media: {df_stats['num_slices'].mean():.1f}")

    if processing_errors:
        print(f"\nERRORI DI PROCESSING:")
        for error in processing_errors:
            print(f" {error['relative_path']}: {error['error']}")

if __name__ == "__main__":
    print("🚀 SCRIPT INTERATTIVO COMPRESSION WITH CROP + TRACKING")
    print("=" * 60)
    print("🆕 NUOVE FUNZIONALITÀ:")
    print(" • 📝 Tracking automatico file già processati")
    print(" • 🔄 Riprendi esecuzione dopo interruzione")
    print(" • 💾 Log automatico di tutti i file processati")
    print(" • ⏯️ Interrompi con Ctrl+C e riprendi quando vuoi")
    print("=" * 60)

    # ⚠️ MODIFICA QUESTI PERCORSI PER IL TUO PROCESSING ⚠️
    
    input_folder = 'F:\\Organoids\\Noyaux\\Cystiques\\Nice'
    output_folder = 'F:\\Organoids\\Cystiques_Nice_Reduce'
    input_folder = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data"
    output_folder = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data_reduced"
    print(f"📂 Input: {input_folder}")
    print(f"📂 Output: {output_folder}")

    if not os.path.exists(input_folder):
        print(f"❌ Cartella input non trovata: {input_folder}")
        print("📝 Modifica le variabili input_folder e output_folder nel main")
    else:
        # Esegui la versione interattiva con tracking
        analyze_and_resize_tiff_volumes_interactive(
            input_folder, 
            output_folder, 
            target_shape=(512, 512), 
            preserve_structure=False
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per visualizzare file TIFF 3D e creare subset con range Z specificato
Basato sullo script check-black-tiff-immediate.py
Autore: Assistant 
Data: Settembre 2025
"""

import os
import numpy as np
from PIL import Image
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import shutil
from pathlib import Path

# Sopprimi warnings per visualizzazione più pulita
warnings.filterwarnings('ignore')

# MODIFICA QUESTO PERCORSO CON LA TUA CARTELLA CONTENENTE I FILE TIFF
FOLDER_PATH = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data"

def setup_folders(base_path):
    """
    Crea le cartelle necessarie per organizzare i file
    Returns: dict con i percorsi delle cartelle create
    """
    folders = {
        'selected_ranges': os.path.join(base_path, 'selected_z_ranges'),
        'processed_log': os.path.join(base_path, 'z_selection_log.txt')
    }

    # Crea le cartelle se non esistono
    for folder_type, folder_path in folders.items():
        if folder_type != 'processed_log':  # Skip del file log
            os.makedirs(folder_path, exist_ok=True)
            print(f"📁 Cartella '{folder_type}' pronta: {folder_path}")

    return folders

def load_processed_files(log_file_path):
    """
    Carica la lista dei file già processati per evitare duplicati
    """
    processed = set()
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        processed.add(line.split('\t')[0])  # Solo il nome file
            print(f"📋 Caricati {len(processed)} file già processati")
        except Exception as e:
            print(f"⚠️ Errore nella lettura del log: {e}")
    else:
        print("📋 Nessun log precedente trovato, inizio da zero")

    return processed

def save_processed_file(log_file_path, filename, action, z_range=None):
    """
    Salva un file nel log dei processati
    """
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            z_info = f"z_{z_range[0]}-{z_range[1]}" if z_range else "skipped"
            f.write(f"{filename}\t{action}\t{z_info}\t{timestamp}\n")
    except Exception as e:
        print(f"⚠️ Errore nel salvare nel log: {e}")

def visualizza_tiff_3d_con_slider(image_path, filename):
    """
    Visualizza il file TIFF 3D con informazioni dettagliate per facilitare la selezione del range Z
    """
    try:
        with Image.open(image_path) as img:
            # Verifica se è un volume 3D multi-frame
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
                print(f" 📏 Range Z disponibile: 0 - {depth-1}")

                # Mostra statistiche per ogni slice per aiutare nella selezione
                print(f"\n 📊 ANTEPRIMA SLICE (per aiutarti nella selezione):")
                print(f" {'Z-Slice':<8} {'Min':<8} {'Max':<8} {'Mean':<10} {'Std':<10} {'Non-Zero':<10}")
                print("-" * 60)

                # Mostra statistiche per alcune slice campione
                sample_indices = []
                if depth <= 10:
                    sample_indices = list(range(depth))
                else:
                    # Mostra primo, ultimo, e alcuni intermedi
                    step = max(1, depth // 8)
                    sample_indices = list(range(0, depth, step))
                    if (depth-1) not in sample_indices:
                        sample_indices.append(depth-1)

                for z in sorted(sample_indices):
                    slice_data = volume_3d[z, :, :]
                    non_zero_count = np.count_nonzero(slice_data)
                    non_zero_percent = (non_zero_count / slice_data.size) * 100
                    print(f" {z:<8} {slice_data.min():<8} {slice_data.max():<8} "
                          f"{slice_data.mean():<10.2f} {slice_data.std():<10.2f} "
                          f"{non_zero_percent:<10.1f}%")

                # Visualizza slice significative
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                # Slice da mostrare
                slice_indices = [
                    0,  # Prima
                    depth//4,  # Primo quarto
                    depth//2,  # Centro
                    3*depth//4,  # Ultimo quarto  
                    depth-1,  # Ultima
                    depth//3  # Un'altra intermedia
                ]

                slice_titles = ['Prima (0)', f'Quarto ({depth//4})', f'Centro ({depth//2})', 
                              f'3/4 ({3*depth//4})', f'Ultima ({depth-1})', f'Terzo ({depth//3})']

                for i, (z_idx, title) in enumerate(zip(slice_indices, slice_titles)):
                    if z_idx < depth:
                        row, col = divmod(i, 3)
                        slice_data = volume_3d[z_idx, :, :]
                        vmin, vmax = slice_data.min(), slice_data.max()

                        im = axes[row, col].imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
                        axes[row, col].set_title(f'{title}\nRange: [{vmin}, {vmax}]')
                        axes[row, col].axis('off')

                        if vmax > vmin:
                            plt.colorbar(im, ax=axes[row, col], shrink=0.6)

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

                print(f"\n 📊 RIEPILOGO VOLUME:")
                print(f" • Totale slice: {depth} (indici: 0-{depth-1})")
                print(f" • Risoluzione XY: {width}×{height}")
                print(f" • Range intensità: [{vol_min}, {vol_max}]")
                print(f" • Statistiche: {vol_mean:.3f} ± {vol_std:.3f}")

                return volume_3d, depth

            else:
                print(f" ⚠️ Il file '{filename}' non è un volume 3D multi-frame")
                print(f" 📋 Informazioni: {img.size}, mode={img.mode}")
                return None, 0

    except Exception as e:
        print(f" ❌ Errore nella visualizzazione: {str(e)}")
        return None, 0

def chiedi_range_z(max_depth):
    """
    Chiede all'utente di specificare il range Z da estrarre
    """
    print(f"\n🎯 SELEZIONE RANGE Z")
    print(f" Depth disponibile: 0 - {max_depth-1} (totale: {max_depth} slice)")
    print()
    print(f" 💡 Esempi di input:")
    print(f"  - '0,{max_depth-1}' = tutto il volume")
    print(f"  - '10,50' = dalle slice 10 alla 50")
    print(f"  - '{max_depth//4},{3*max_depth//4}' = metà centrale del volume")
    print()

    while True:
        try:
            user_input = input(f" 👉 Inserisci il range (inizio,fine) o 'skip' per saltare: ").strip()

            if user_input.lower() in ['skip', 's', 'salta']:
                print(" ⏭️ File saltato")
                return None

            if ',' in user_input:
                start_str, end_str = user_input.split(',', 1)
                start_z = int(start_str.strip())
                end_z = int(end_str.strip())

                # Validazione
                if start_z < 0 or end_z >= max_depth:
                    print(f" ⚠️ Range fuori dai limiti! Usa valori tra 0 e {max_depth-1}")
                    continue

                if start_z >= end_z:
                    print(f" ⚠️ L'inizio deve essere minore della fine!")
                    continue

                num_slices = end_z - start_z + 1
                print(f" ✅ Range selezionato: {start_z} - {end_z} ({num_slices} slice)")
                return (start_z, end_z)
            else:
                print(" ⚠️ Formato non valido! Usa 'inizio,fine' (es: 10,50)")

        except ValueError:
            print(" ⚠️ Inserisci numeri validi!")
        except KeyboardInterrupt:
            print("\n 🛑 Interruzione da tastiera")
            return None
        except EOFError:
            print("\n 🛑 Input terminato")
            return None

def crea_subset_volume(volume_3d, z_range, original_filename, output_folder):
    """
    Crea un nuovo file TIFF con il subset del volume specificato
    """
    start_z, end_z = z_range

    # Estrai il subset
    subset_volume = volume_3d[start_z:end_z+1, :, :]
    subset_depth = subset_volume.shape[0]

    print(f" 🔪 Estrazione subset: slice {start_z}-{end_z} ({subset_depth} slice)")

    # Crea il nome del file di output
    base_name, ext = os.path.splitext(original_filename)
    output_filename = f"{base_name}_z{start_z}-{end_z}{ext}"
    output_path = os.path.join(output_folder, output_filename)

    try:
        # Salva il subset come nuovo file TIFF multi-frame
        if subset_depth == 1:
            # Volume singolo - salva come immagine 2D
            subset_image = Image.fromarray(subset_volume[0])
            subset_image.save(output_path)
        else:
            # Volume multi-frame - salva come stack TIFF
            frames = [Image.fromarray(frame) for frame in subset_volume]
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                compression='tiff_lzw'  # Compressione per ridurre dimensione file
            )

        # Verifica che il file sia stato creato
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f" ✅ File creato: {output_filename}")
            print(f" 📏 Dimensioni: {subset_volume.shape} ({file_size_mb:.2f} MB)")
            print(f" 📁 Percorso: {output_path}")
            return True, output_filename
        else:
            print(f" ❌ Errore: il file non è stato creato")
            return False, None

    except Exception as e:
        print(f" ❌ Errore nella creazione del file: {str(e)}")
        return False, None

def process_tiff_files_with_z_selection(folder_path):
    """
    Processa i file TIFF permettendo selezione range Z e creazione subset
    """
    # Verifica che la cartella esista
    if not os.path.exists(folder_path):
        print(f"❌ Errore: La cartella '{folder_path}' non esiste!")
        return None

    # Setup delle cartelle
    folders = setup_folders(folder_path)

    # Carica i file già processati
    processed_files = load_processed_files(folders['processed_log'])

    # Trova tutti i file TIFF nella cartella principale
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tiff_files = []
    for pattern in tiff_patterns:
        tiff_files.extend(glob.glob(os.path.join(folder_path, pattern)))

    if not tiff_files:
        print(f"⚠️ Nessun file TIFF trovato nella cartella '{folder_path}'")
        return None

    # Filtra i file già processati
    remaining_files = [f for f in tiff_files if os.path.basename(f) not in processed_files]

    print(f"🔍 File TIFF totali: {len(tiff_files)}")
    print(f"📋 File già processati: {len(processed_files)}")
    print(f"🎯 File rimanenti da processare: {len(remaining_files)}")

    if not remaining_files:
        print("✅ Tutti i file sono già stati processati!")
        return {
            'total_files': len(tiff_files),
            'processed_count': len(processed_files),
            'remaining_count': 0,
            'completed': True
        }

    print("="*70)
    print("🎯 MODALITÀ SELEZIONE RANGE Z ATTIVA")
    print(" • Ogni file TIFF 3D verrà visualizzato")
    print(" • Potrai specificare un range di slice Z da estrarre")
    print(" • Verrà creato un nuovo file con solo quelle slice")
    print(" • I nuovi file saranno salvati in 'selected_z_ranges'")
    print("="*70)

    stats = {
        'processed': 0,
        'ranges_created': 0,
        'skipped': 0,
        'errors': 0,
        'total_new_files': 0
    }

    for i, tiff_file in enumerate(remaining_files):
        filename = os.path.basename(tiff_file)
        current_pos = len(processed_files) + i + 1
        total_files = len(tiff_files)

        print(f"\n[{current_pos:3d}/{total_files}] 📄 Processando: {filename}")
        print("-" * 50)

        try:
            # Visualizza il file e ottieni il volume
            print("🖼️ Caricamento e visualizzazione del volume 3D...")
            volume_3d, depth = visualizza_tiff_3d_con_slider(tiff_file, filename)

            if volume_3d is None or depth == 0:
                print("⏭️ File saltato (non è un volume 3D valido)")
                save_processed_file(folders['processed_log'], filename, 'skipped_not_3d')
                stats['skipped'] += 1
                stats['processed'] += 1
                continue

            # Chiedi il range Z
            z_range = chiedi_range_z(depth)

            if z_range is None:
                print("⏭️ File saltato dall'utente")
                save_processed_file(folders['processed_log'], filename, 'skipped_by_user')
                stats['skipped'] += 1
            else:
                # Crea il subset
                print(f"\n🔧 Creazione subset dal volume originale...")
                success, new_filename = crea_subset_volume(
                    volume_3d, z_range, filename, folders['selected_ranges']
                )

                if success:
                    stats['ranges_created'] += 1
                    stats['total_new_files'] += 1
                    save_processed_file(folders['processed_log'], filename, 'range_created', z_range)
                    print(f" 🎉 Subset creato con successo!")
                else:
                    stats['errors'] += 1
                    save_processed_file(folders['processed_log'], filename, 'error_creating_range', z_range)

            stats['processed'] += 1

            # Mostra progresso
            print(f"\n📊 Progresso: {stats['processed']} file processati")
            print(f" ✅ Range creati: {stats['ranges_created']} | ⏭️ Saltati: {stats['skipped']} | ❌ Errori: {stats['errors']}")

        except KeyboardInterrupt:
            print("\n🛑 Interruzione da tastiera")
            break
        except Exception as e:
            print(f"❌ Errore nel processamento del file {filename}: {str(e)}")
            stats['errors'] += 1
            stats['processed'] += 1

    # Report finale
    print("\n" + "="*60)
    print("📊 REPORT FINALE SESSIONE")
    print("="*60)
    print(f"📁 Cartella processata: {folder_path}")
    print(f"🎯 File processati in questa sessione: {stats['processed']}")
    print(f"✅ Nuovi file creati con range Z: {stats['ranges_created']}")
    print(f"⏭️ File saltati: {stats['skipped']}")
    print(f"❌ Errori: {stats['errors']}")
    print(f"\n📁 Cartella output: {folders['selected_ranges']}")
    print(f"📄 Log dettagliato: {folders['processed_log']}")

    remaining_after_session = len(tiff_files) - len(processed_files) - stats['processed']
    if remaining_after_session > 0:
        print(f"\n⏳ File ancora da processare: {remaining_after_session}")
        print("💡 Puoi riprendere eseguendo nuovamente lo script")
    else:
        print("\n🎉 TUTTI I FILE SONO STATI PROCESSATI!")

    print("="*60)

    return {
        'total_files': len(tiff_files),
        'session_processed': stats['processed'],
        'ranges_created': stats['ranges_created'],
        'remaining': remaining_after_session,
        'stats': stats,
        'folders': folders
    }

def main():
    """
    Funzione principale dello script
    """
    print("🚀 VISUALIZZATORE TIFF 3D CON SELEZIONE RANGE Z")
    print("=" * 50)
    print("🎯 Funzionalità:")
    print(" • Visualizza file TIFF 3D con statistiche dettagliate")
    print(" • Permette selezione di range Z specifici")
    print(" • Crea nuovi file con solo le slice selezionate")

    # Verifica che il percorso sia stato modificato
    if FOLDER_PATH == "/path/to/your/tiff/folder":
        print("\n❌ ERRORE: Devi modificare la variabile FOLDER_PATH!")
        print("\n🔧 ISTRUZIONI:")
        print("1. Aprire questo file con un editor di testo")
        print("2. Modificare la riga 'FOLDER_PATH = ...' con il percorso corretto")
        print("3. Salvare il file e rieseguire")
        print("\n📝 Esempi di percorsi:")
        print(" Windows: r'C:\\Users\\NomeUtente\\Documenti\\CartellaTiff'")
        print(" Mac/Linux: '/Users/nomeutente/Documenti/CartellaTiff'")
        return

    print(f"\n📁 Cartella da processare: {FOLDER_PATH}")
    print("\n💡 COME FUNZIONA:")
    print(" • Ogni file TIFF 3D viene caricato e visualizzato")
    print(" • Vedi statistiche dettagliate per ogni slice Z")
    print(" • Specifichi il range di slice da estrarre (es: 10,50)")
    print(" • Viene creato un nuovo file con solo quelle slice")
    print(" • I nuovi file vengono salvati in 'selected_z_ranges'")

    print("\n🎯 Cartelle che verranno create:")
    print(" 📁 'selected_z_ranges' - per i file con range Z selezionati")
    print(" 📄 'z_selection_log.txt' - log delle operazioni")

    print("\n🚀 Inizio processamento...\n")

    try:
        # Esegui il processamento
        results = process_tiff_files_with_z_selection(FOLDER_PATH)

        if results:
            print("\n✅ Sessione completata con successo!")
            if results.get('remaining', 0) > 0:
                print("💡 Esegui nuovamente lo script per continuare dai file rimanenti")

    except KeyboardInterrupt:
        print("\n\n🛑 Programma interrotto dall'utente (Ctrl+C)")
        print("💡 Il progresso è stato salvato. Puoi riprendere eseguendo nuovamente lo script")

    except Exception as e:
        print(f"\n\n❌ Errore inaspettato: {str(e)}")

if __name__ == "__main__":
    main()

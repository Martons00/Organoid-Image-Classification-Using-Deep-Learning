#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per visualizzare file TIFF 3D con viste ortogonali e creare subset con range Z specificato
Versione aggiornata con tifffile per salvataggio ottimizzato
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

def visualizza_tiff_3d(image_path, filename):
    """
    Visualizza il file TIFF con viste ortogonali per facilitare la selezione del range Z
    """
    try:
        with Image.open(image_path) as img:
            # Carica tutto il volume se è multi-frame
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"  📚 Caricamento volume 3D ({img.n_frames} frame)...")

                # Carica tutti i frame in un array 3D
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    frame = np.array(img)
                    frames.append(frame)

                # Crea array 3D: (depth, height, width)
                volume_3d = np.stack(frames, axis=0)
                depth, height, width = volume_3d.shape
                print(f"  📦 Dimensioni volume: {depth}x{height}x{width} (D×H×W)")

                # Calcola slice centrali per ogni piano
                z_center = depth // 2
                y_center = height // 2
                x_center = width // 2

                # Estrai i tre piani ortogonali
                slice_xy = volume_3d[z_center, :, :]  # Piano XY (slice lungo Z)
                slice_yz = volume_3d[:, y_center, :]  # Piano YZ (slice lungo Y)
                slice_xz = volume_3d[:, :, x_center]  # Piano XZ (slice lungo X)

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

                print(f"  📊 Statistiche volume completo:")
                print(f"     Range: [{vol_min}, {vol_max}]")
                print(f"     Media: {vol_mean:.3f} ± {vol_std:.3f}")
                print(f"     Slice centrali: XY={z_center}, YZ={x_center}, XZ={y_center}")
                print(f"     Range Z disponibile: 0 - {depth-1}")

                return volume_3d, depth

            else:
                # File 2D singolo
                img_array = np.array(img)
                if len(img_array.shape) == 2:
                    # Immagine 2D grayscale
                    height, width = img_array.shape

                    # Crea figura con 3 subplot: immagine + istogramma + profili
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    # Immagine principale
                    vmin, vmax = img_array.min(), img_array.max()
                    im = axes[0].imshow(img_array, cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0].set_title(f'Immagine 2D\n{height}×{width}\nmin={vmin}, max={vmax}')
                    axes[0].set_xlabel('X (Width)')
                    axes[0].set_ylabel('Y (Height)')
                    if vmax > vmin:
                        plt.colorbar(im, ax=axes[0], shrink=0.8)

                    # Istogramma intensità
                    axes[1].hist(img_array.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                    axes[1].set_title('Istogramma Intensità')
                    axes[1].set_xlabel('Valore Pixel')
                    axes[1].set_ylabel('Frequenza')
                    axes[1].grid(True, alpha=0.3)

                    # Profili centrali
                    center_row = height // 2
                    center_col = width // 2
                    axes[2].plot(img_array[center_row, :], 'b-', label=f'Riga centrale ({center_row})', alpha=0.8)
                    axes[2].plot(img_array[:, center_col], 'r-', label=f'Colonna centrale ({center_col})', alpha=0.8)
                    axes[2].set_title('Profili Centrali')
                    axes[2].set_xlabel('Posizione Pixel')
                    axes[2].set_ylabel('Intensità')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)

                    plt.suptitle(f'Immagine 2D: {filename}', fontsize=14, fontweight='bold')

                elif len(img_array.shape) == 3:
                    # Immagine RGB o multi-canale
                    height, width, channels = img_array.shape
                    fig, axes = plt.subplots(1, min(3, channels), figsize=(6 * min(3, channels), 6))
                    if channels == 1:
                        axes = [axes]

                    for i in range(min(3, channels)):
                        channel_data = img_array[:, :, i]
                        vmin, vmax = channel_data.min(), channel_data.max()
                        im = axes[i].imshow(channel_data, cmap='gray', vmin=vmin, vmax=vmax)
                        axes[i].set_title(f'Canale {i}\nmin={vmin}, max={vmax}')
                        axes[i].axis('off')
                        if vmax > vmin:
                            plt.colorbar(im, ax=axes[i], shrink=0.8)

                    plt.suptitle(f'Immagine Multi-canale: {filename}\n'
                                f'Shape: {height}×{width}×{channels}',
                                fontsize=14, fontweight='bold')

                print(f"  📊 Immagine 2D: shape={img_array.shape}")
                print(f"     Range: [{img_array.min()}, {img_array.max()}]")
                print(f"     Media: {img_array.mean():.3f} ± {img_array.std():.3f}")

                plt.tight_layout()
                plt.show()

                return None, 0

    except Exception as e:
        print(f"  ❌ Errore nella visualizzazione: {str(e)}")
        # Fallback: prova a mostrare almeno le info base
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'n_frames'):
                    print(f"  📋 Info file: {img.n_frames} frame, mode={img.mode}")
                else:
                    print(f"  📋 Info file: {img.size}, mode={img.mode}")
        except:
            print("  ❌ Impossibile leggere anche le informazioni base del file")

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
    Crea un nuovo file TIFF con il subset del volume specificato usando tifffile
    """
    start_z, end_z = z_range

    # Estrai il subset
    subset_volume = volume_3d[start_z:end_z+1, :, :]
    subset_depth = subset_volume.shape[0]

    print(f" 🔪 Estrazione subset: slice {start_z}-{end_z} ({subset_depth} slice)")

    # Crea il nome del file di output
    base_name, ext = os.path.splitext(original_filename)
    output_filename = f"{base_name}_z{start_z}-{end_z}.tif"  # Forza estensione .tif
    output_path = os.path.join(output_folder, output_filename)

    try:
        # Salva il subset usando tifffile
        import tifffile as tiff
        tiff.imwrite(str(output_path), subset_volume)

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
    print("🎯 MODALITÀ SELEZIONE RANGE Z CON VISTE ORTOGONALI")
    print(" • Ogni file TIFF 3D viene visualizzato con piani ortogonali")
    print(" • XY (assiale), YZ (sagittale), XZ (coronale)")
    print(" • I nuovi file vengono salvati usando tifffile per massima compatibilità")
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
            print("🖼️ Caricamento e visualizzazione del volume 3D con viste ortogonali...")
            volume_3d, depth = visualizza_tiff_3d(tiff_file, filename)

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
                print(f"\n🔧 Creazione subset dal volume originale con tifffile...")
                success, new_filename = crea_subset_volume(
                    volume_3d, z_range, filename, folders['selected_ranges']
                )

                if success:
                    stats['ranges_created'] += 1
                    stats['total_new_files'] += 1
                    save_processed_file(folders['processed_log'], filename, 'range_created', z_range)
                    print(f" 🎉 Subset creato con successo usando tifffile!")
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
    print(f"💾 I file sono stati salvati usando tifffile per massima compatibilità")

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
    print("🚀 VISUALIZZATORE TIFF 3D CON VISTE ORTOGONALI E TIFFFILE")
    print("=" * 58)
    print("🎯 Funzionalità:")
    print(" • Visualizza file TIFF 3D con viste ortogonali (XY, YZ, XZ)")
    print(" • Mostra slice centrali di tutti e tre i piani anatomici")
    print(" • Permette selezione di range Z specifici")
    print(" • Salva usando tifffile per massima compatibilità")

    # Verifica che il percorso sia stato modificato
    if FOLDER_PATH == "/path/to/your/tiff/folder":
        print("\n❌ ERRORE: Devi modificare la variabile FOLDER_PATH!")
        print("\n🔧 ISTRUZIONI:")
        print("1. Aprire questo file con un editor di testo")
        print("2. Modificare la riga 'FOLDER_PATH = ...' con il percorso corretto")
        print("3. Installare tifffile: pip install tifffile")
        print("4. Salvare il file e rieseguire")
        print("\n📝 Esempi di percorsi:")
        print(" Windows: r'C:\\Users\\NomeUtente\\Documenti\\CartellaTiff'")
        print(" Mac/Linux: '/Users/nomeutente/Documenti/CartellaTiff'")
        return

    # Verifica che tifffile sia installato
    try:
        import tifffile
        print("✅ tifffile disponibile")
    except ImportError:
        print("\n❌ ERRORE: tifffile non installato!")
        print("\n🔧 SOLUZIONE:")
        print("Installa tifffile con: pip install tifffile")
        print("Poi riesegui lo script")
        return

    print(f"\n📁 Cartella da processare: {FOLDER_PATH}")
    print("\n💡 COME FUNZIONA:")
    print(" • Ogni file TIFF 3D viene caricato e visualizzato")
    print(" • Vedi 3 viste ortogonali: XY (assiale), YZ (sagittale), XZ (coronale)")
    print(" • Le viste mostrano le slice centrali di ogni piano")
    print(" • Specifichi il range di slice Z da estrarre (es: 10,50)")
    print(" • Il nuovo file viene salvato usando tifffile (.tif)")

    print("\n🔧 VANTAGGI DI TIFFFILE:")
    print(" 📦 Gestione automatica di volumi 2D e 3D")
    print(" 🔧 Metadati TIFF accurati e completi")
    print(" ⚡ Performance ottimizzate per grandi volumi")
    print(" 💾 Massima compatibilità con software di imaging")

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

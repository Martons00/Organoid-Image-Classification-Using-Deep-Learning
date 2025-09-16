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

FOLDER_PATH = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data"

def setup_folders(base_path):
    """
    Crea le cartelle necessarie per organizzare i file
    Returns: dict con i percorsi delle cartelle create
    """
    folders = {
        'check': os.path.join(base_path, 'check'),
        'black': os.path.join(base_path, 'black'),
        'processed_log': os.path.join(base_path, 'processed_files.txt')
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
                        processed.add(line)
            print(f"📋 Caricati {len(processed)} file già processati")
        except Exception as e:
            print(f"⚠️ Errore nella lettura del log: {e}")
    else:
        print("📋 Nessun log precedente trovato, inizio da zero")

    return processed

def save_processed_file(log_file_path, filename, action):
    """
    Salva un file nel log dei processati
    """
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{filename}\t{action}\t{timestamp}\n")
    except Exception as e:
        print(f"⚠️ Errore nel salvare nel log: {e}")

def visualizza_tiff_3d(image_path, filename):
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

                print(f"  📊 Statistiche volume completo:")
                print(f"     Range: [{vol_min}, {vol_max}]")
                print(f"     Media: {vol_mean:.3f} ± {vol_std:.3f}")
                print(f"     Slice centrali: XY={z_center}, YZ={x_center}, XZ={y_center}")

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

def chiedi_decisione_utente(filename):
    """
    Chiede all'utente cosa fare con il file corrente
    """
    print(f"\n❓ DECISIONE RICHIESTA per: '{filename}'")
    print("   Il file è stato mostrato sopra")
    print()
    print("   📋 OPZIONI DISPONIBILI:")
    print("   [B] - File NERO (sposta in cartella 'black')")
    print("   [C] - File da CONTROLLARE/SALTARE (sposta in cartella 'check')")
    print("   [N] - File NORMALE (lascia nella cartella originale)")
    print("   [Q] - ESCI dal programma")
    print()

    while True:
        try:
            scelta = input("   👉 La tua scelta [B/C/N/Q]: ").strip().upper()

            if scelta in ['B', 'BLACK', 'NERO']:
                print("   🖤 File classificato come NERO")
                return 'black'
            elif scelta in ['C', 'CHECK', 'SALTA', 'SKIP']:
                print("   🔍 File spostato in CHECK")
                return 'check'
            elif scelta in ['N', 'NORMAL', 'NORMALE', '']:
                print("   ✅ File classificato come NORMALE")
                return 'normal'
            elif scelta in ['Q', 'QUIT', 'EXIT']:
                print("   🛑 Uscita richiesta")
                return 'quit'
            else:
                print("   ⚠️  Scelta non valida. Usa: B (nero), C (check), N (normale), Q (esci)")

        except KeyboardInterrupt:
            print("\n   🛑 Interruzione da tastiera - uscita")
            return 'quit'
        except EOFError:
            print("\n   🛑 Input terminato - uscita")
            return 'quit'

def sposta_file_immediatamente(file_path, destination_folder, action):
    """
    Sposta il file immediatamente nella cartella di destinazione
    """
    filename = os.path.basename(file_path)
    destination_path = os.path.join(destination_folder, filename)

    try:
        # Controlla se esiste già un file con lo stesso nome
        if os.path.exists(destination_path):
            # Crea un nome alternativo con timestamp
            base_name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%H%M%S')
            new_filename = f"{base_name}_{timestamp}{ext}"
            destination_path = os.path.join(destination_folder, new_filename)
            print(f"   🔄 File rinominato in: {new_filename}")

        # Sposta il file
        shutil.move(file_path, destination_path)
        print(f"   ✅ File spostato in: {os.path.basename(destination_folder)}")
        return True, os.path.basename(destination_path)

    except Exception as e:
        print(f"   ❌ Errore nello spostamento: {str(e)}")
        return False, None

def process_files_with_immediate_move(folder_path):
    """
    Processa i file uno alla volta con spostamento immediato
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
        print(f"⚠️  Nessun file TIFF trovato nella cartella '{folder_path}'")
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
    print("🎯 MODALITÀ PROCESSAMENTO IMMEDIATO ATTIVA")
    print("   • Ogni file verrà mostrato e potrai decidere immediatamente")
    print("   • I file verranno spostati subito nelle cartelle appropriate")
    print("   • Puoi interrompere e riprendere in qualsiasi momento")
    print("   • Il progresso viene salvato automaticamente")
    print("="*70)

    stats = {
        'processed': 0,
        'moved_to_black': 0,
        'moved_to_check': 0,
        'kept_normal': 0,
        'errors': 0
    }

    for i, tiff_file in enumerate(remaining_files):
        filename = os.path.basename(tiff_file)
        current_pos = len(processed_files) + i + 1
        total_files = len(tiff_files)

        print(f"\n[{current_pos:3d}/{total_files}] 📄 Processando: {filename}")
        print("-" * 50)

        try:
            # Mostra il file
            print("🖼️  Visualizzazione del file...")
            visualizza_tiff_3d(tiff_file, filename)

            # Chiedi decisione
            decision = chiedi_decisione_utente(filename)

            if decision == 'quit':
                print("\n🛑 Uscita richiesta dall'utente")
                break

            elif decision == 'black':
                success, new_name = sposta_file_immediatamente(tiff_file, folders['black'], 'black')
                if success:
                    stats['moved_to_black'] += 1
                    save_processed_file(folders['processed_log'], filename, 'black')
                else:
                    stats['errors'] += 1

            elif decision == 'check':
                success, new_name = sposta_file_immediatamente(tiff_file, folders['check'], 'check')
                if success:
                    stats['moved_to_check'] += 1
                    save_processed_file(folders['processed_log'], filename, 'check')
                else:
                    stats['errors'] += 1

            elif decision == 'normal':
                stats['kept_normal'] += 1
                save_processed_file(folders['processed_log'], filename, 'normal')
                print("   📁 File lasciato nella cartella originale")

            stats['processed'] += 1

            # Mostra progresso
            print(f"\n📊 Progresso: {stats['processed']} file processati")
            print(f"   🖤 Black: {stats['moved_to_black']} | 🔍 Check: {stats['moved_to_check']} | ✅ Normal: {stats['kept_normal']} | ❌ Errori: {stats['errors']}")

        except Exception as e:
            print(f"❌ Errore nel processamento del file {filename}: {str(e)}")
            stats['errors'] += 1

    # Report finale
    print("\n" + "="*60)
    print("📊 REPORT FINALE SESSIONE")
    print("="*60)
    print(f"📁 Cartella processata: {folder_path}")
    print(f"🎯 File processati in questa sessione: {stats['processed']}")
    print(f"🖤 Spostati in 'black': {stats['moved_to_black']}")
    print(f"🔍 Spostati in 'check': {stats['moved_to_check']}")
    print(f"✅ Lasciati come 'normal': {stats['kept_normal']}")
    print(f"❌ Errori: {stats['errors']}")
    print(f"\n📋 Totale file processati complessivamente: {len(processed_files) + stats['processed']}/{len(tiff_files)}")

    remaining_after_session = len(tiff_files) - len(processed_files) - stats['processed']
    if remaining_after_session > 0:
        print(f"⏳ File ancora da processare: {remaining_after_session}")
        print("💡 Puoi riprendere eseguendo nuovamente lo script")
    else:
        print("🎉 TUTTI I FILE SONO STATI PROCESSATI!")

    print("="*60)

    return {
        'total_files': len(tiff_files),
        'session_processed': stats['processed'],
        'total_processed': len(processed_files) + stats['processed'],
        'remaining': remaining_after_session,
        'stats': stats,
        'folders': folders
    }

def main():
    """
    Funzione principale dello script con processamento immediato
    """
    print("🚀 SCANNER TIFF CON SPOSTAMENTO IMMEDIATO")
    print("="*45)
    print("🎯 Modalità: Processamento file singolo con spostamento istantaneo")

    # Verifica che il percorso sia stato modificato
    if FOLDER_PATH == "/path/to/your/tiff/folder":
        print("\n❌ ERRORE: Devi modificare la variabile FOLDER_PATH!")
        print("\n🔧 ISTRUZIONI:")
        print("1. Aprire questo file con un editor di testo")
        print("2. Modificare la riga 'FOLDER_PATH = ...' con il percorso corretto")
        print("3. Salvare il file e rieseguire")
        print("\n📝 Esempi di percorsi:")
        print("   Windows: r'C:\\Users\\NomeUtente\\Documenti\\CartellaTiff'")
        print("   Mac/Linux: '/Users/nomeutente/Documenti/CartellaTiff'")
        return

    print(f"\n📁 Cartella da processare: {FOLDER_PATH}")
    print("\n💡 COME FUNZIONA:")
    print("   • Ogni file viene mostrato singolarmente")
    print("   • Decidi immediatamente se è nero, da controllare o normale")
    print("   • Il file viene spostato subito nella cartella appropriata")
    print("   • Il progresso viene salvato automaticamente")
    print("   • Puoi interrompere e riprendere in qualsiasi momento")
    print("\n🎯 Cartelle che verranno create:")
    print("   📁 'black' - per file confermati come neri")
    print("   📁 'check' - per file da controllare/saltati")
    print("   📄 'processed_files.txt' - log dei file processati")
    print("\n🚀 Inizio processamento...\n")

    try:
        # Esegui il processamento con spostamento immediato
        results = process_files_with_immediate_move(FOLDER_PATH)

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

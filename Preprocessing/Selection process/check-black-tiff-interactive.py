import os
import numpy as np
from PIL import Image
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import os
import shutil
from datetime import datetime
from pathlib import Path

# Sopprimi warnings per visualizzazione più pulita
warnings.filterwarnings('ignore')
 
FOLDER_PATH = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data"  

def visualizza_tiff_3d(image_path, filename):

    try:
        with Image.open(image_path) as img:
            # Carica tutto il volume se è multi-frame
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"    📚 Caricamento volume 3D ({img.n_frames} frame)...")
                
                # Carica tutti i frame in un array 3D
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    frame = np.array(img)
                    frames.append(frame)
                
                # Crea array 3D: (depth, height, width)
                volume_3d = np.stack(frames, axis=0)
                depth, height, width = volume_3d.shape
                
                print(f"    📦 Dimensioni volume: {depth}x{height}x{width} (D×H×W)")
                
                # Calcola slice centrali per ogni piano
                z_center = depth // 2
                y_center = height // 2  
                x_center = width // 2
                
                # Estrai i tre piani ortogonali
                slice_xy = volume_3d[z_center, :, :]      # Piano XY (slice lungo Z)
                slice_yz = volume_3d[:, y_center, :]      # Piano YZ (slice lungo Y)  
                slice_xz = volume_3d[:, :, x_center]      # Piano XZ (slice lungo X)
                
                # Crea figura con 3 subplot
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Piano XY (assiale)
                vmin_xy, vmax_xy = slice_xy.min(), slice_xy.max()
                im1 = axes[0].imshow(slice_xy, cmap='gray', vmin=vmin_xy, vmax=vmax_xy)
                axes[0].set_title(f'Piano XY (Assiale)\\nSlice Z={z_center}/{depth-1}\\nmin={vmin_xy}, max={vmax_xy}')
                axes[0].set_xlabel('X (Width)')
                axes[0].set_ylabel('Y (Height)')
                if vmax_xy > vmin_xy:
                    plt.colorbar(im1, ax=axes[0], shrink=0.8)
                
                # Piano YZ (sagittale)  
                vmin_yz, vmax_yz = slice_yz.min(), slice_yz.max()
                im2 = axes[1].imshow(slice_yz, cmap='gray', vmin=vmin_yz, vmax=vmax_yz, aspect='auto')
                axes[1].set_title(f'Piano YZ (Sagittale)\\nSlice X={x_center}/{width-1}\\nmin={vmin_yz}, max={vmax_yz}')
                axes[1].set_xlabel('Z (Depth)')
                axes[1].set_ylabel('Y (Height)')
                if vmax_yz > vmin_yz:
                    plt.colorbar(im2, ax=axes[1], shrink=0.8)
                
                # Piano XZ (coronale)
                vmin_xz, vmax_xz = slice_xz.min(), slice_xz.max()
                im3 = axes[2].imshow(slice_xz, cmap='gray', vmin=vmin_xz, vmax=vmax_xz, aspect='auto')
                axes[2].set_title(f'Piano XZ (Coronale)\\nSlice Y={y_center}/{height-1}\\nmin={vmin_xz}, max={vmax_xz}')
                axes[2].set_xlabel('X (Width)')
                axes[2].set_ylabel('Z (Depth)')
                if vmax_xz > vmin_xz:
                    plt.colorbar(im3, ax=axes[2], shrink=0.8)
                
                # Statistiche del volume completo
                vol_min, vol_max = volume_3d.min(), volume_3d.max()
                vol_mean = volume_3d.mean()
                vol_std = volume_3d.std()
                
                plt.suptitle(f'Volume 3D: {filename}\\n'
                           f'Shape: {depth}×{height}×{width} | '
                           f'Range: [{vol_min}, {vol_max}] | '
                           f'Mean: {vol_mean:.2f} ± {vol_std:.2f}', 
                           fontsize=14, fontweight='bold')
                
                print(f"    📊 Statistiche volume completo:")
                print(f"        Range: [{vol_min}, {vol_max}]")
                print(f"        Media: {vol_mean:.3f} ± {vol_std:.3f}")
                print(f"        Slice centrali: XY={z_center}, YZ={x_center}, XZ={y_center}")
                
            else:
                # File 2D singolo - mostra con informazioni aggiuntive
                img_array = np.array(img)
                
                if len(img_array.shape) == 2:
                    # Immagine 2D grayscale
                    height, width = img_array.shape
                    
                    # Crea figura con 3 subplot: immagine + istogramma + profili
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Immagine principale
                    vmin, vmax = img_array.min(), img_array.max()
                    im = axes[0].imshow(img_array, cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0].set_title(f'Immagine 2D\\n{height}×{width}\\nmin={vmin}, max={vmax}')
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
                    
                    # Profili centrali (riga e colonna centrale)
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
                        axes[i].set_title(f'Canale {i}\\nmin={vmin}, max={vmax}')
                        axes[i].axis('off')
                        
                        if vmax > vmin:
                            plt.colorbar(im, ax=axes[i], shrink=0.8)
                    
                    plt.suptitle(f'Immagine Multi-canale: {filename}\\n'
                               f'Shape: {height}×{width}×{channels}', 
                               fontsize=14, fontweight='bold')
                
                print(f"    📊 Immagine 2D: shape={img_array.shape}")
                print(f"        Range: [{img_array.min()}, {img_array.max()}]")
                print(f"        Media: {img_array.mean():.3f} ± {img_array.std():.3f}")
            
            plt.tight_layout()
            plt.show()
                
    except Exception as e:
        print(f"    ❌ Errore nella visualizzazione: {str(e)}")
        
        # Fallback: prova a mostrare almeno le info base
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'n_frames'):
                    print(f"    📋 Info file: {img.n_frames} frame, mode={img.mode}")
                else:
                    print(f"    📋 Info file: {img.size}, mode={img.mode}")
        except:
            print("    ❌ Impossibile leggere anche le informazioni base del file")

def chiedi_conferma_utente(filename):

    print(f"\\n❓ DECISIONE RICHIESTA per: '{filename}'")
    print("   Il file risulta tecnicamente completamente nero (tutti pixel = 0)")
    print()
    print("   📋 OPZIONI DISPONIBILI:")
    print("   [N] - È veramente NERO (mantieni nella lista file neri)")
    print("   [E] - È un ERRORE/CORROTTO (sposta nella lista errori)")  
    print("   [S] - SALTA questo file (non classificare in nessuna lista)")
    print("   [Q] - ESCI dal programma")
    print()
    
    while True:
        try:
            scelta = input("   👉 La tua scelta [N/E/S/Q]: ").strip().upper()
            
            if scelta in ['N', 'NERO', '']:
                print("   ✅ File confermato come NERO")
                return 'nero'
            elif scelta in ['E', 'ERRORE']:
                print("   ❌ File classificato come ERRORE")
                return 'errore'
            elif scelta in ['S', 'SALTA', 'SKIP']:
                print("   ⏭️  File saltato")
                return 'salta'
            elif scelta in ['Q', 'QUIT', 'EXIT']:
                print("   🛑 Uscita richiesta")
                return 'quit'
            else:
                print("   ⚠️  Scelta non valida. Usa: N (nero), E (errore), S (salta), Q (esci)")
                
        except KeyboardInterrupt:
            print("\\n   🛑 Interruzione da tastiera - uscita")
            return 'quit'
        except EOFError:
            print("\\n   🛑 Input terminato - uscita")
            return 'quit'

def check_if_completely_black_interactive(image_path):
    filename = os.path.basename(image_path)
    
    try:
        with Image.open(image_path) as img:
            # Controllo tecnico se è nero
            print(f"    🖼️  Mostro il file per verifica visiva...")
            
            # Visualizza il file
            visualizza_tiff_3d(image_path, filename)
            
            # Chiedi conferma all'utente
            user_choice = chiedi_conferma_utente(filename)
            return True, user_choice
    except Exception as e:
        print(f"    ❌ Errore nella lettura del file: {str(e)}")
        print(f"    💡 Questo potrebbe indicare un file corrotto")
        return None, 'errore'

def find_black_tiff_volumes_interactive(folder_path):
    # Verifica che la cartella esista
    if not os.path.exists(folder_path):
        print(f"❌ Errore: La cartella '{folder_path}' non esiste!")
        return {}
    
    # Trova tutti i file TIFF nella cartella
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tiff_files = []
    
    for pattern in tiff_patterns:
        tiff_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        # Decommentare per cercare anche nelle sottocartelle:
        # tiff_files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))
    
    if not tiff_files:
        print(f"⚠️  Nessun file TIFF trovato nella cartella '{folder_path}'")
        return {}
    
    print(f"🔍 Trovati {len(tiff_files)} file TIFF da analizzare...")
    print("="*70)
    print("🎯 MODALITÀ INTERATTIVA ATTIVA")
    print("   • I file rilevati come neri verranno visualizzati")
    print("   • Potrai confermare se sono veramente neri o errori")
    print("   • Usa Ctrl+C per interrompere in qualsiasi momento")
    print("="*70)
    
    black_files = []           # File confermati come neri
    processed_files = []       # File normali (non neri)
    error_files = []          # File con errori o riclassificati come errori
    skipped_files = []        # File saltati dall'utente
    user_quit = False         # Flag per uscita utente
    
    for i, tiff_file in enumerate(tiff_files):
        if user_quit:
            print("\\n🛑 Analisi interrotta dall'utente")
            break
            
        filename = os.path.basename(tiff_file)
        print(f"\\n[{i+1:3d}/{len(tiff_files)}] 📄 Analizzando: {filename}")
        
        is_black, classification = check_if_completely_black_interactive(tiff_file)
        
        # Gestisci la risposta dell'utente
        if classification == 'quit':
            user_quit = True
            break
        elif is_black is None or classification == 'errore':
            error_files.append(filename)
            print(f"    📝 ➜ Aggiunto alla lista ERRORI")
        elif is_black and classification == 'nero':
            black_files.append(filename)
            print(f"    📝 ➜ Confermato nella lista FILE NERI")
        elif is_black and classification == 'salta':
            skipped_files.append(filename)
            print(f"    📝 ➜ File saltato (non classificato)")
        else:
            processed_files.append(filename)
            print(f"    📝 ➜ File normale (contiene dati validi)")
    
    return {
        'total_files': len(tiff_files),
        'black_files': black_files,
        'processed_files': processed_files,
        'error_files': error_files,
        'skipped_files': skipped_files,
        'folder_path': folder_path,
        'user_quit': user_quit,
        'analyzed_files': i + 1 if not user_quit else i
    }

def generate_interactive_report(results):
    """
    Genera un report dettagliato dei risultati dell'analisi interattiva
    
    Args:
        results (dict): Risultati dell'analisi
    """
    if not results:
        return
        
    print("\\n" + "="*70)
    print("📊 REPORT FINALE - ANALISI INTERATTIVA VOLUMI TIFF")
    print("="*70)
    
    print(f"📁 Cartella analizzata: {results['folder_path']}")
    print(f"📋 File TIFF totali trovati: {results['total_files']}")
    print(f"🔍 File effettivamente analizzati: {results['analyzed_files']}")
    
    if results.get('user_quit'):
        print(f"🛑 ⚠️  ANALISI INTERROTTA DALL'UTENTE")
        print(f"    Restano da analizzare: {results['total_files'] - results['analyzed_files']} file")
    
    print(f"\\n📈 RISULTATI CLASSIFICAZIONE:")
    print(f"✅ File normali (con dati): {len(results['processed_files'])}")
    print(f"🚨 File neri confermati: {len(results['black_files'])}")  
    print(f"❌ File errori/corrotti: {len(results['error_files'])}")
    print(f"⏭️  File saltati: {len(results['skipped_files'])}")
    
    if results['black_files']:
        print(f"\\n🚨 FILE CONFERMATI COME COMPLETAMENTE NERI ({len(results['black_files'])}):")
        print("-" * 50)
        for i, filename in enumerate(results['black_files'], 1):
            print(f"  {i:2d}. {filename}")
            
        print(f"\\n📋 LISTA NOMI FILE NERI (per copia/incolla):")
        print("-" * 50)
        for filename in results['black_files']:
            print(filename)
    else:
        print("\\n✅ OTTIMO: Nessun volume nero confermato!")
    
    if results['error_files']:
        print(f"\\n❌ FILE CLASSIFICATI COME ERRORI/CORROTTI ({len(results['error_files'])}):")
        print("-" * 50)
        for filename in results['error_files']:
            print(f"    • {filename}")
    
    if results['skipped_files']:
        print(f"\\n⏭️  FILE SALTATI DURANTE L'ANALISI ({len(results['skipped_files'])}):")
        print("-" * 50) 
        for filename in results['skipped_files']:
            print(f"    • {filename}")
    
    print("\\n" + "="*70)
    
    return results

def save_interactive_results(results, output_file="analisi_tiff_interattiva.txt"):

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"REPORT ANALISI INTERATTIVA FILE TIFF\\n")
            f.write(f"======================================\\n\\n")
            f.write(f"Cartella analizzata: {results['folder_path']}\\n")
            f.write(f"Data e ora analisi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"File TIFF totali trovati: {results['total_files']}\\n")
            f.write(f"File analizzati: {results['analyzed_files']}\\n")
            
            if results.get('user_quit'):
                f.write(f"NOTA: Analisi interrotta dall'utente\\n")
                
            f.write(f"\\nRISULTATI CLASSIFICAZIONE:\\n")
            f.write(f"- File normali: {len(results['processed_files'])}\\n")
            f.write(f"- File neri confermati: {len(results['black_files'])}\\n")
            f.write(f"- File errori/corrotti: {len(results['error_files'])}\\n")
            f.write(f"- File saltati: {len(results['skipped_files'])}\\n")
            f.write("-" * 50 + "\\n\\n")
            
            if results['black_files']:
                f.write("FILE CONFERMATI COME NERI:\\n")
                f.write("-" * 30 + "\\n")
                for i, filename in enumerate(results['black_files'], 1):
                    f.write(f"{i:3d}. {filename}\\n")
                f.write("\\n")
            
            if results['error_files']:
                f.write("FILE CLASSIFICATI COME ERRORI:\\n")
                f.write("-" * 30 + "\\n")
                for filename in results['error_files']:
                    f.write(f"    {filename}\\n")
                f.write("\\n")
            
            if results['skipped_files']:
                f.write("FILE SALTATI:\\n")
                f.write("-" * 30 + "\\n")
                for filename in results['skipped_files']:
                    f.write(f"    {filename}\\n")
        
        print(f"\\n💾 Report dettagliato salvato in: {output_file}")
        return True
        
    except Exception as e:
        print(f"\\n❌ Errore nel salvare il report: {str(e)}")
        return False
    

def sposta_file_neri(results, nome_sottocartella=None):
    """
    Sposta i file identificati come neri in una sottocartella specifica
    
    Args:
        results (dict): Dizionario dei risultati dell'analisi TIFF
        nome_sottocartella (str, optional): Nome personalizzato per la sottocartella
                                          Se None, usa un nome automatico
    
    Returns:
        dict: Risultati dell'operazione di spostamento
    """
    
    if not results or not results.get('black_files'):
        print("📋 Nessun file nero da spostare")
        return {
            'success': True,
            'moved_files': [],
            'errors': [],
            'subfolder_path': None,
            'message': 'Nessun file da spostare'
        }
    
    # Genera nome sottocartella se non specificato
    if nome_sottocartella is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_sottocartella = f"black_tiff_files_{timestamp}"
    
    # Crea il percorso della sottocartella
    cartella_origine = results['folder_path']
    sottocartella_path = os.path.join(cartella_origine, nome_sottocartella)
    
    print(f"📁 Creazione sottocartella: {nome_sottocartella}")
    print(f"🎯 Percorso completo: {sottocartella_path}")
    
    moved_files = []
    error_files = []
    
    try:
        # Crea la sottocartella se non esiste
        os.makedirs(sottocartella_path, exist_ok=True)
        print(f"✅ Sottocartella creata con successo")
        
        print(f"\\n🚚 Inizio spostamento di {len(results['black_files'])} file...")
        print("="*50)
        
        for i, filename in enumerate(results['black_files'], 1):
            print(f"[{i:2d}/{len(results['black_files'])}] 📦 Spostando: {filename}")
            
            # Percorsi sorgente e destinazione
            percorso_origine = os.path.join(cartella_origine, filename)
            percorso_destinazione = os.path.join(sottocartella_path, filename)
            
            try:
                # Verifica che il file sorgente esista
                if not os.path.exists(percorso_origine):
                    print(f"    ⚠️  File non trovato: {percorso_origine}")
                    error_files.append({
                        'filename': filename,
                        'error': 'File non trovato',
                        'source_path': percorso_origine
                    })
                    continue
                
                # Verifica che non esista già un file con lo stesso nome nella destinazione
                if os.path.exists(percorso_destinazione):
                    print(f"    ⚠️  File già esistente nella destinazione")
                    # Crea un nome alternativo con timestamp
                    base_name, ext = os.path.splitext(filename)
                    timestamp_file = datetime.now().strftime('%H%M%S')
                    new_filename = f"{base_name}_{timestamp_file}{ext}"
                    percorso_destinazione = os.path.join(sottocartella_path, new_filename)
                    print(f"    🔄 Rinominato in: {new_filename}")
                
                # Sposta il file
                shutil.move(percorso_origine, percorso_destinazione)
                moved_files.append({
                    'original_name': filename,
                    'new_name': os.path.basename(percorso_destinazione),
                    'source_path': percorso_origine,
                    'destination_path': percorso_destinazione
                })
                print(f"    ✅ Spostato con successo")
                
            except PermissionError as e:
                print(f"    ❌ Errore permessi: {str(e)}")
                error_files.append({
                    'filename': filename,
                    'error': f'Errore permessi: {str(e)}',
                    'source_path': percorso_origine
                })
                
            except Exception as e:
                print(f"    ❌ Errore generico: {str(e)}")
                error_files.append({
                    'filename': filename,
                    'error': f'Errore generico: {str(e)}',
                    'source_path': percorso_origine
                })
        
        # Crea un file di log nella sottocartella
        log_file_path = os.path.join(sottocartella_path, "spostamento_log.txt")
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"LOG SPOSTAMENTO FILE TIFF NERI\\n")
            log_file.write(f"================================\\n\\n")
            log_file.write(f"Data e ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            log_file.write(f"Cartella origine: {cartella_origine}\\n")
            log_file.write(f"Sottocartella: {nome_sottocartella}\\n")
            log_file.write(f"Totale file da spostare: {len(results['black_files'])}\\n")
            log_file.write(f"File spostati con successo: {len(moved_files)}\\n")
            log_file.write(f"File con errori: {len(error_files)}\\n")
            log_file.write(f"\\nFILE SPOSTATI:\\n")
            log_file.write("-" * 30 + "\\n")
            for file_info in moved_files:
                log_file.write(f"- {file_info['original_name']} -> {file_info['new_name']}\\n")
            
            if error_files:
                log_file.write(f"\\nFILE CON ERRORI:\\n")
                log_file.write("-" * 30 + "\\n")
                for error_info in error_files:
                    log_file.write(f"- {error_info['filename']}: {error_info['error']}\\n")
        
        print(f"\\n📄 Log salvato in: spostamento_log.txt")
        
    except Exception as e:
        print(f"❌ Errore nella creazione della sottocartella: {str(e)}")
        return {
            'success': False,
            'moved_files': [],
            'errors': [{'error': f'Errore creazione cartella: {str(e)}'}],
            'subfolder_path': sottocartella_path,
            'message': f'Errore: {str(e)}'
        }
    
    # Report finale
    print("\\n" + "="*50)
    print("📊 REPORT SPOSTAMENTO")
    print("="*50)
    print(f"📁 Sottocartella creata: {nome_sottocartella}")
    print(f"✅ File spostati con successo: {len(moved_files)}")
    print(f"❌ File con errori: {len(error_files)}")
    
    if moved_files:
        print(f"\\n📦 FILE SPOSTATI:")
        for file_info in moved_files:
            if file_info['original_name'] != file_info['new_name']:
                print(f"  • {file_info['original_name']} -> {file_info['new_name']}")
            else:
                print(f"  • {file_info['original_name']}")
    
    if error_files:
        print(f"\\n❌ ERRORI:")
        for error_info in error_files:
            print(f"  • {error_info['filename']}: {error_info['error']}")
    
    print("="*50)
    
    return {
        'success': len(error_files) == 0,
        'moved_files': moved_files,
        'errors': error_files,
        'subfolder_path': sottocartella_path,
        'subfolder_name': nome_sottocartella,
        'total_files': len(results['black_files']),
        'successful_moves': len(moved_files),
        'failed_moves': len(error_files),
        'message': f'Spostati {len(moved_files)}/{len(results["black_files"])} file'
    }

def sposta_file_per_categoria(results, crea_sottocartelle=True):
    """
    Sposta tutti i file problematici (neri, errori) in sottocartelle separate
    
    Args:
        results (dict): Dizionario dei risultati dell'analisi TIFF
        crea_sottocartelle (bool): Se True crea sottocartelle separate per ogni categoria
    
    Returns:
        dict: Risultati dell'operazione di spostamento per tutte le categorie
    """
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cartella_origine = results['folder_path']
    
    risultati_spostamento = {
        'black_files': None,
        'error_files': None,
        'timestamp': timestamp,
        'origin_folder': cartella_origine
    }
    
    print("🗂️  SPOSTAMENTO MULTI-CATEGORIA")
    print("="*40)
    
    # Sposta file neri
    if results.get('black_files'):
        print(f"\\n📦 Spostamento file NERI ({len(results['black_files'])} file)")
        nome_sottocartella_neri = f"black_files_{timestamp}" if crea_sottocartelle else "black_files"
        risultati_neri = sposta_file_neri(results, nome_sottocartella_neri)
        risultati_spostamento['black_files'] = risultati_neri
    
    # Sposta file errori
    if results.get('error_files'):
        print(f"\\n❌ Spostamento file ERRORI ({len(results['error_files'])} file)")
        
        # Crea dizionario simulato per file errori
        results_errori = {
            'black_files': results['error_files'],  # Usa la stessa logica
            'folder_path': cartella_origine
        }
        
        nome_sottocartella_errori = f"error_files_{timestamp}" if crea_sottocartelle else "error_files"
        risultati_errori = sposta_file_neri(results_errori, nome_sottocartella_errori)
        risultati_spostamento['error_files'] = risultati_errori
    
    # Report finale multi-categoria
    print("\\n" + "="*60)
    print("📊 REPORT FINALE SPOSTAMENTO MULTI-CATEGORIA")
    print("="*60)
    
    if risultati_spostamento['black_files']:
        neri = risultati_spostamento['black_files']
        print(f"📦 File neri: {neri['successful_moves']}/{neri['total_files']} spostati")
        print(f"   📁 Sottocartella: {neri['subfolder_name']}")
    
    if risultati_spostamento['error_files']:
        errori = risultati_spostamento['error_files']
        print(f"❌ File errori: {errori['successful_moves']}/{errori['total_files']} spostati")
        print(f"   📁 Sottocartella: {errori['subfolder_name']}")
    
    print("="*60)
    
    return risultati_spostamento

def main():
    """
    Funzione principale dello script interattivo
    """
    print("🚀 SCANNER INTERATTIVO VOLUMI TIFF NERI")
    print("="*45)
    print("🎯 Modalità: INTERATTIVA con visualizzazione")
    
    # Verifica che il percorso sia stato modificato
    if FOLDER_PATH == "/path/to/your/tiff/folder":
        print("\\n❌ ERRORE: Devi modificare la variabile FOLDER_PATH!")
        print("\\n🔧 ISTRUZIONI:")
        print("1. Aprire questo file con un editor di testo")
        print("2. Modificare la riga 'FOLDER_PATH = ...' con il percorso corretto")
        print("3. Salvare il file e rieseguire")
        print("\\n📝 Esempi di percorsi:")
        print("   Windows: r'C:\\\\Users\\\\NomeUtente\\\\Documenti\\\\CartellaTiff'")
        print("   Mac/Linux: '/Users/nomeutente/Documenti/CartellaTiff'")
        return
    
    print(f"\\n📁 Cartella da analizzare: {FOLDER_PATH}")
    print("\\n💡 COME FUNZIONA:")
    print("   • Ogni file rilevato come nero verrà mostrato")
    print("   • Deciderai se è veramente nero o un errore")
    print("   • Potrai saltare file o interrompere quando vuoi")
    print("\\n🚀 Inizio analisi...\\n")
    
    try:
        # Esegui l'analisi interattiva
        results = find_black_tiff_volumes_interactive(FOLDER_PATH)
        
        if not results:
            return
        
        # Genera il report
        generate_interactive_report(results)
        
        # Salva automaticamente i risultati
        if results['total_files'] > 0:
            save_interactive_results(results)
            sposta_file_neri(results)

            
        # Riepilogo finale
        print(f"\\n🎯 ANALISI COMPLETATA")
        if results.get('user_quit'):
            print(f"   ⚠️  Interrotta dall'utente ({results['analyzed_files']}/{results['total_files']} file analizzati)")
        else:
            print(f"   ✅ Tutti i {results['total_files']} file sono stati processati")
            
        print(f"   🚨 File neri confermati: {len(results['black_files'])}")
        print(f"   ❌ File con errori: {len(results['error_files'])}")
        
    except KeyboardInterrupt:
        print("\\n\\n🛑 Programma interrotto dall'utente (Ctrl+C)")
    except Exception as e:
        print(f"\\n\\n❌ Errore inaspettato: {str(e)}")

if __name__ == "__main__":
    main()
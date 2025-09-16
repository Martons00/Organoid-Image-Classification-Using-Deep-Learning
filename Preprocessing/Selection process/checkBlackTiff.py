
import os
import numpy as np
from PIL import Image
import glob
from datetime import datetime

FOLDER_PATH = "/Users/raffaelemartone34gmail.com/Desktop/Politecnico/Tesi/Repo/Organoid-Image-Classification-Using-Deep-Learning/data"  

def check_if_completely_black(image_path):
    """
    Controlla se un file TIFF è completamente nero (tutti i pixel hanno valore 0)
    
    Args:
        image_path (str): Percorso del file TIFF
        
    Returns:
        bool: True se l'immagine è completamente nera, False altrimenti
        None: Se c'è stato un errore nella lettura
    """
    try:
        with Image.open(image_path) as img:
            # Gestione di TIFF multi-frame (stack/volumi 3D)
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"    📚 File multi-frame rilevato ({img.n_frames} frame)")
                # Per TIFF stack, controlla tutti i frame
                for frame_idx in range(img.n_frames):
                    img.seek(frame_idx)
                    frame_array = np.array(img)
                    
                    # Se anche solo un frame non è nero, l'intero volume non è nero
                    if not np.all(frame_array == 0):
                        return False
                return True
            else:
                # Per TIFF singolo
                img_array = np.array(img)
                return np.all(img_array == 0)
                
    except Exception as e:
        print(f"    ❌ Errore: {str(e)}")
        return None

def find_black_tiff_volumes(folder_path):
    """
    Scansiona una cartella per trovare tutti i file TIFF completamente neri
    
    Args:
        folder_path (str): Percorso della cartella da scansionare
        
    Returns:
        dict: Dizionario con i risultati dell'analisi
    """
    
    # Verifica che la cartella esista
    if not os.path.exists(folder_path):
        print(f"❌ Errore: La cartella '{folder_path}' non esiste!")
        return {}
    
    # Trova tutti i file TIFF nella cartella
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tiff_files = []
    
    for pattern in tiff_patterns:
        tiff_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        # Decommentare la riga seguente per cercare anche nelle sottocartelle
        # tiff_files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))
    
    if not tiff_files:
        print(f"⚠️  Nessun file TIFF trovato nella cartella '{folder_path}'")
        return {}
    
    print(f"🔍 Trovati {len(tiff_files)} file TIFF da analizzare...")
    print("="*70)
    
    black_files = []
    processed_files = []
    error_files = []
    
    for i, tiff_file in enumerate(tiff_files):
        filename = os.path.basename(tiff_file)
        print(f"[{i+1:3d}/{len(tiff_files)}] 📄 {filename}")
        
        result = check_if_completely_black(tiff_file)
        
        if result is None:
            error_files.append(filename)
            print(f"    ⚠️  Errore nella lettura")
        elif result:
            black_files.append(filename)
            print(f"    🚨 VOLUME COMPLETAMENTE NERO!")
        else:
            processed_files.append(filename)
            print(f"    ✅ Volume OK (contiene dati)")
    
    return {
        'total_files': len(tiff_files),
        'black_files': black_files,
        'processed_files': processed_files,
        'error_files': error_files,
        'folder_path': folder_path
    }

def generate_report(results):
    """
    Genera un report dettagliato dei risultati
    
    Args:
        results (dict): Risultati dell'analisi
    """
    if not results:
        return
        
    print("\n" + "="*70)
    print("📊 REPORT FINALE - ANALISI VOLUMI TIFF")
    print("="*70)
    
    print(f"📁 Cartella analizzata: {results['folder_path']}")
    print(f"📋 File totali processati: {results['total_files']}")
    print(f"❌ File con errori: {len(results['error_files'])}")
    print(f"✅ File validi: {len(results['processed_files'])}")
    print(f"🚨 File completamente neri: {len(results['black_files'])}")
    
    if results['black_files']:
        print(f"\n🚨 ATTENZIONE: {len(results['black_files'])} VOLUMI COMPLETAMENTE NERI RILEVATI:")
        print("-" * 50)
        for i, filename in enumerate(results['black_files'], 1):
            print(f"  {i:2d}. {filename}")
            
        print(f"\n📋 LISTA NOMI FILE NERI (per copia/incolla):")
        print("-" * 50)
        for filename in results['black_files']:
            print(filename)
            
    else:
        print("\n✅ OTTIMO: Nessun volume completamente nero trovato!")
    
    if results['error_files']:
        print(f"\n❌ File con errori di lettura ({len(results['error_files'])}):")
        print("-" * 50)
        for filename in results['error_files']:
            print(f"    • {filename}")
    
    print("\n" + "="*70)
    
    return results['black_files']  # Restituisce la lista per uso programmatico

def save_black_files_list(results, output_file="file_tiff_neri.txt"):
    """
    Salva la lista dei file neri in un file di testo
    
    Args:
        results (dict): Risultati dell'analisi
        output_file (str): Nome del file di output
    """
    if results['black_files']:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"LISTA FILE TIFF COMPLETAMENTE NERI\n")
                f.write(f"====================================\n\n")
                f.write(f"Cartella analizzata: {results['folder_path']}\n")
                f.write(f"Data e ora analisi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Totale file TIFF analizzati: {results['total_files']}\n")
                f.write(f"File completamente neri trovati: {len(results['black_files'])}\n")
                f.write(f"File con errori: {len(results['error_files'])}\n")
                f.write("-" * 50 + "\n\n")
                
                f.write("ELENCO FILE NERI:\n")
                f.write("-" * 20 + "\n")
                for i, filename in enumerate(results['black_files'], 1):
                    f.write(f"{i:3d}. {filename}\n")
                
                if results['error_files']:
                    f.write(f"\n\nFILE CON ERRORI:\n")
                    f.write("-" * 20 + "\n")
                    for filename in results['error_files']:
                        f.write(f"    {filename}\n")
            
            print(f"\n💾 Lista salvata nel file: {output_file}")
            return True
            
        except Exception as e:
            print(f"\n❌ Errore nel salvare il file: {str(e)}")
            return False
    else:
        print("\n💡 Nessun file da salvare (nessun volume nero trovato)")
        return False

def main():
    """
    Funzione principale dello script
    """
    print("🚀 SCANNER VOLUMI TIFF COMPLETAMENTE NERI")
    print("="*45)
    
    # Verifica che il percorso sia stato modificato
    if FOLDER_PATH == "/path/to/your/tiff/folder":
        print("❌ ERRORE: Devi modificare la variabile FOLDER_PATH!")
        print("\n🔧 ISTRUZIONI:")
        print("1. Aprire questo file con un editor di testo")
        print("2. Modificare la riga 'FOLDER_PATH = ...' con il percorso corretto")
        print("3. Salvare il file e rieseguire")
        print("\n📝 Esempi di percorsi:")
        print("   Windows: r'C:\\Users\\NomeUtente\\Documenti\\CartellaTiff'")
        print("   Mac/Linux: '/Users/nomeutente/Documenti/CartellaTiff'")
        return
    
    print(f"📁 Cartella da analizzare: {FOLDER_PATH}")
    print()
    
    # Esegui l'analisi
    results = find_black_tiff_volumes(FOLDER_PATH)
    
    if not results:
        return
    
    # Genera il report
    black_files = generate_report(results)
    
    # Salva automaticamente la lista se ci sono file neri
    if black_files:
        save_black_files_list(results)
        
    print(f"\n🎯 RISULTATO FINALE:")
    if black_files:
        print(f"   Trovati {len(black_files)} file completamente neri su {results['total_files']} analizzati")
    else:
        print(f"   Tutti i {results['total_files']} file TIFF analizzati contengono dati validi")
    
    return results

if __name__ == "__main__":
    main()
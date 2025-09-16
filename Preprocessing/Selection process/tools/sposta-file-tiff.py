import os
import shutil
from datetime import datetime
from pathlib import Path

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

# Esempi di utilizzo
def esempi_utilizzo():
    """
    Esempi di come utilizzare le funzioni di spostamento
    """
    
    print("📋 ESEMPI DI UTILIZZO:")
    print("="*30)
    
    print("\\n1️⃣ SPOSTAMENTO SOLO FILE NERI:")
    print("   # Dopo aver eseguito l'analisi")
    print("   results = find_black_tiff_volumes_interactive('/path/to/folder')")
    print("   ")
    print("   # Sposta con nome automatico")
    print("   move_result = sposta_file_neri(results)")
    print("   ")
    print("   # Sposta con nome personalizzato")
    print("   move_result = sposta_file_neri(results, 'volumi_problematici_2025')")
    
    print("\\n2️⃣ SPOSTAMENTO MULTI-CATEGORIA:")
    print("   # Sposta file neri E errori in sottocartelle separate")
    print("   move_result = sposta_file_per_categoria(results)")
    print("   ")
    print("   # Usa nomi fissi (senza timestamp)")
    print("   move_result = sposta_file_per_categoria(results, crea_sottocartelle=False)")
    
    print("\\n3️⃣ CONTROLLO RISULTATI:")
    print("   if move_result['success']:")
    print("       print(f'✅ Spostati {move_result[\"successful_moves\"]} file')")
    print("   else:")
    print("       print(f'❌ Errori: {len(move_result[\"errors\"])}')")
    
    print("\\n4️⃣ STRUTTURA CARTELLE RISULTANTE:")
    print("   cartella_originale/")
    print("   ├── file_normali.tiff")
    print("   ├── black_tiff_files_20250916_140530/")
    print("   │   ├── file_nero_1.tiff")
    print("   │   ├── file_nero_2.tiff")
    print("   │   └── spostamento_log.txt")
    print("   └── error_files_20250916_140530/")
    print("       ├── file_corrotto_1.tiff")
    print("       └── spostamento_log.txt")

if __name__ == "__main__":
    esempi_utilizzo()
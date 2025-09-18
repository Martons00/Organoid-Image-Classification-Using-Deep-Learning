import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
from skimage.transform import resize
import tifffile as tiff
import os
import glob
import pandas as pd
from pathlib import Path  # AGGIUNTO


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


def analyze_and_resize_tiff_volumes(input_folder, output_folder, target_shape=(512, 512), preserve_structure=True):
    """
    Analizza e ridimensiona tutti i file .tif in una cartella e sottocartelle, raccogliendo statistiche
    e generando grafici di analisi salvati nella cartella di output.
    
    Args:
        input_folder (str): Percorso cartella input contenente i file .tif
        output_folder (str): Percorso cartella output per file ridimensionati
        target_shape (tuple): Dimensioni target per il ridimensionamento (y, x)
        preserve_structure (bool): Se True mantiene la struttura delle sottocartelle nell'output
    """
    
    # Crea cartelle se non esistono
    os.makedirs(output_folder, exist_ok=True)
    plots_folder = os.path.join(output_folder, "analysis_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # MODIFICATO: Trova tutti i file .tif ricorsivamente nelle sottocartelle
    input_path = Path(input_folder)
    tif_files = list(input_path.glob("**/*.tif"))  # Ricerca ricorsiva
    tif_files_str = [str(f) for f in tif_files]  # Converti in stringhe per compatibilità
    
    if not tif_files:
        print(f"Nessun file .tif trovato nella cartella {input_folder} e sue sottocartelle")
        return
    
    print(f"Trovati {len(tif_files)} file .tif da processare (incluse sottocartelle)...")
    
    # AGGIUNTO: Mostra struttura delle cartelle trovate
    subdirs_found = set()
    for tif_file in tif_files:
        relative_path = tif_file.relative_to(input_path)
        if len(relative_path.parts) > 1:  # Ha sottocartelle
            subdir = relative_path.parent
            subdirs_found.add(str(subdir))
    
    if subdirs_found:
        print(f"Sottocartelle trovate: {sorted(subdirs_found)}")
    
    # Liste per raccogliere statistiche
    volume_stats = []
    processing_errors = []
    
    # Processa ogni file
    for i, tif_path in enumerate(tif_files_str):
        try:
            # AGGIUNTO: Calcola percorso relativo per mantenere struttura
            tif_path_obj = Path(tif_path)
            relative_path = tif_path_obj.relative_to(input_path)
            
            # Carica volume
            volume = io.imread(tif_path)
            volume = normalize_uint16_to_uint8(volume)

            #CROPPING PHASE
            volume = volume[:,300:750,300:750]
            
            if len(volume.shape) < 3:
                print(f"SKIP: {relative_path} - Non è un volume 3D")
                continue
            
            # MODIFICATO: Raccogli statistiche (incluso percorso relativo)
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
                'file_size_mb': os.path.getsize(tif_path) / (1024**2)
            }
            
            # Ridimensiona volume
            resized_volume = resize(
                volume, 
                (volume.shape[0], target_shape[0], target_shape[1]), 
                order=1, 
                preserve_range=True, 
                anti_aliasing=True
            ).astype(volume.dtype)
            
            # AGGIUNTO: Determina percorso di output
            if preserve_structure:
                output_path = Path(output_folder) / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)  # Crea sottocartelle se necessario
            else:
                # Salva tutto nella cartella principale, rinomina se conflitti
                output_filename = f"{relative_path.parent}_{tif_path_obj.name}".replace(os.sep, "_")
                if str(relative_path.parent) == ".":  # File nella cartella root
                    output_filename = tif_path_obj.name
                output_path = Path(output_folder) / output_filename
            
            # Salva volume ridimensionato
            tiff.imwrite(str(output_path), resized_volume)
            
            # MODIFICATO: Aggiorna statistiche post-processing
            file_stats['output_path'] = str(output_path.relative_to(Path(output_folder)))
            file_stats['resized_shape'] = resized_volume.shape
            file_stats['resized_file_size_mb'] = os.path.getsize(str(output_path)) / (1024**2)
            file_stats['size_reduction_percent'] = (1 - file_stats['resized_file_size_mb'] / file_stats['file_size_mb']) * 100
            
            volume_stats.append(file_stats)
            
            # Progress update ogni 10 file
            if (i + 1) % 10 == 0 or (i + 1) == len(tif_files):
                progress = ((i + 1) / len(tif_files)) * 100
                print(f"Processati: {i + 1}/{len(tif_files)} file ({progress:.1f}%)")
                
        except Exception as e:
            # MODIFICATO: Gestione errori con percorso relativo
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
    _generate_analysis_plots(df_stats, plots_folder)
    _print_summary_statistics(df_stats, processing_errors, input_folder, output_folder)
    
    # Salva statistiche in CSV
    stats_csv_path = os.path.join(output_folder, "volume_statistics.csv")
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
    
    # 2. Dimensioni originali (scatter plot height vs width)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_stats['original_width'], df_stats['original_height'], alpha=0.6, color='coral')
    plt.xlabel('Larghezza Originale (pixel)')
    plt.ylabel('Altezza Originale (pixel)')
    plt.title('Distribuzione delle Dimensioni Originali')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_folder, 'original_dimensions.png'), dpi=300, bbox_inches='tight')
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
    
    # 4. Box plot dei valori di intensità
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Min values
    axes[0,0].boxplot(df_stats['min_value'])
    axes[0,0].set_title('Distribuzione Valori Minimi')
    axes[0,0].set_ylabel('Intensità')
    
    # Max values
    axes[0,1].boxplot(df_stats['max_value'])
    axes[0,1].set_title('Distribuzione Valori Massimi')
    axes[0,1].set_ylabel('Intensità')
    
    # Mean values
    axes[1,0].boxplot(df_stats['mean_value'])
    axes[1,0].set_title('Distribuzione Valori Medi')
    axes[1,0].set_ylabel('Intensità')
    
    # Std values
    axes[1,1].boxplot(df_stats['std_value'])
    axes[1,1].set_title('Distribuzione Deviazione Standard')
    axes[1,1].set_ylabel('Intensità')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'intensity_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlazione numero slice vs dimensione file
    plt.figure(figsize=(10, 6))
    plt.scatter(df_stats['num_slices'], df_stats['file_size_mb'], alpha=0.6, color='purple')
    plt.xlabel('Numero di Slice')
    plt.ylabel('Dimensione File (MB)')
    plt.title('Correlazione: Numero Slice vs Dimensione File')
    plt.grid(True, alpha=0.3)
    
    # Aggiungi linea di regressione
    z = np.polyfit(df_stats['num_slices'], df_stats['file_size_mb'], 1)
    p = np.poly1d(z)
    plt.plot(df_stats['num_slices'], p(df_stats['num_slices']), "r--", alpha=0.8)
    
    plt.savefig(os.path.join(plots_folder, 'slices_vs_filesize.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # AGGIUNTO: 6. Distribuzione file per sottocartella (se ci sono sottocartelle)
    if 'subfolder' in df_stats.columns and len(df_stats['subfolder'].unique()) > 1:
        plt.figure(figsize=(12, 6))
        subfolder_counts = df_stats['subfolder'].value_counts()
        subfolder_counts.plot(kind='bar', color='mediumpurple', alpha=0.7)
        plt.xlabel('Sottocartella')
        plt.ylabel('Numero di File')
        plt.title('Distribuzione File per Sottocartella')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, 'files_per_subfolder.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # AGGIUNTO: 7. Box plot numero slice per sottocartella
        unique_subfolders = df_stats['subfolder'].unique()
        if len(unique_subfolders) > 1:
            plt.figure(figsize=(12, 6))
            data_by_subfolder = [df_stats[df_stats['subfolder'] == sf]['num_slices'] for sf in unique_subfolders]
            plt.boxplot(data_by_subfolder, labels=unique_subfolders)
            plt.xlabel('Sottocartella')
            plt.ylabel('Numero di Slice')
            plt.title('Distribuzione Numero Slice per Sottocartella')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, 'slices_per_subfolder.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Grafici salvati in: {plots_folder}")


def _print_summary_statistics(df_stats, processing_errors, input_folder, output_folder):
    """Stampa statistiche riassuntive del processing"""
    
    print("\n" + "="*60)
    print("STATISTICHE RIASSUNTIVE DEL PROCESSING")
    print("="*60)
    
    print(f"File processati con successo: {len(df_stats)}")
    print(f"File con errori: {len(processing_errors)}")
    
    # AGGIUNTO: Statistiche per sottocartelle
    if 'subfolder' in df_stats.columns:
        subfolders = df_stats['subfolder'].unique()
        if len(subfolders) > 1:
            print(f"\nSTATISTICHE PER SOTTOCARTELLE:")
            for subfolder in sorted(subfolders):
                sf_data = df_stats[df_stats['subfolder'] == subfolder]
                print(f"  {subfolder}: {len(sf_data)} file, "
                      f"slice range: {sf_data['num_slices'].min()}-{sf_data['num_slices'].max()}")
    
    print(f"\nNUMERO DI SLICE (GLOBALE):")
    print(f"  Min: {df_stats['num_slices'].min()}")
    print(f"  Max: {df_stats['num_slices'].max()}")
    print(f"  Media: {df_stats['num_slices'].mean():.1f}")
    print(f"  Mediana: {df_stats['num_slices'].median():.1f}")
    
    print(f"\nDIMENSIONI ORIGINALI:")
    print(f"  Altezza - Min: {df_stats['original_height'].min()}, Max: {df_stats['original_height'].max()}")
    print(f"  Larghezza - Min: {df_stats['original_width'].min()}, Max: {df_stats['original_width'].max()}")
    
    print(f"\nINTENSITÀ PIXEL:")
    print(f"  Range valori - Min: {df_stats['min_value'].min()}, Max: {df_stats['max_value'].max()}")
    print(f"  Media generale: {df_stats['mean_value'].mean():.2f}")
    
    # MODIFICATO: Calcolo dimensioni cartelle (ricorsivo)
    total_input_size = sum(f.stat().st_size for f in Path(input_folder).rglob('*') if f.is_file())
    total_output_size = sum(f.stat().st_size for f in Path(output_folder).rglob('*.tif') if f.is_file())
    
    print(f"\nDIMENSIONI CARTELLE (RICORSIVE):")
    print(f"  Input: {total_input_size / (1024**3):.2f} GB ({total_input_size / (1024**2):.1f} MB)")
    print(f"  Output: {total_output_size / (1024**3):.2f} GB ({total_output_size / (1024**2):.1f} MB)")
    print(f"  Riduzione totale: {(1 - total_output_size / total_input_size) * 100:.1f}%")
    
    if processing_errors:
        print(f"\nERRORI DI PROCESSING:")
        for error in processing_errors:
            print(f"  {error['relative_path']}: {error['error']}")


if __name__ == "__main__":
    # Esempio di utilizzo
    
    input_folder = 'F:\\Organoids\\Noyaux\\Cystiques\\Nice'
    output_folder = 'F:\\Organoids\\Cystiques_Nice_Reduce'
    # MODIFICATO: Aggiunto parametro preserve_structure
    analyze_and_resize_tiff_volumes(input_folder, output_folder, target_shape=(512, 512), preserve_structure=False)

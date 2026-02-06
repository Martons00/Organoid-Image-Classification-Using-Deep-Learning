import os
import json
import tifffile
from pathlib import Path
from datetime import datetime

def extract_metadata(tif_path):
    """Estrae metadati da file .tif"""
    with tifffile.TiffFile(tif_path) as tif:
        # Numero di layer (z-stack)
        n_layers = len(tif.pages)
        
        # Risoluzione (width x height)
        first_page = tif.pages[0]
        resolution = f"{first_page.imagewidth}x{first_page.imagelength}"
        
        # Format (uint8 o uint16)
        dtype = str(first_page.dtype)
        if "uint8" in dtype:
            format_type = "uint8"
        elif "uint16" in dtype:
            format_type = "uint16"
        else:
            format_type = dtype
    
    return n_layers, resolution, format_type

def extract_lab_from_filename(filename):
    """Estrae lab dal nome del file"""
    filename_lower = filename.lower()
    if "paris" in filename_lower:
        return "Metatox (Paris)"
    elif "nice" in filename_lower:
        return "IPMC (Nice)"
    elif "noyau" in filename_lower:
        return "IPMC (Nice)"
    else:
        return "Unknown"

def extract_magnification(filename):
    """Estrae magnification dal nome del file"""
    if "x20" in filename.lower() or "20x" in filename.lower():
        return "20x"
    elif "x40" in filename.lower() or "40x" in filename.lower():
        return "40x"  # Default
    else:
        return "20x"

def create_organoid_database(folder_path, output_file="organoid_db.json"):
    """Crea database JSON con metadati di tutti i .tif nella cartella"""
    
    database = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_samples": 0
        },
        "samples": []
    }
    
    tif_files = sorted(Path(folder_path).glob("**/*.tif"))
    
    for idx, tif_path in enumerate(tif_files, start=1):
        filename = tif_path.name
        
        try:
            n_layers, resolution, format_type = extract_metadata(tif_path)
            lab = extract_lab_from_filename(filename)
            magnification = extract_magnification(filename)
            
            sample = {
                "id": f"ORG_{idx:04d}",
                "filename": filename,
                "lab": lab,
                "magnification": magnification,
                "format": format_type,
                "N_layer": n_layers,
                "resolution": resolution,
                "file_size_mb": tif_path.stat().st_size / (1024**2),
                "path": str(tif_path)
            }
            
            database["samples"].append(sample)
            print(f"[{idx}] {filename} → {sample['id']}")
            
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
    
    database["metadata"]["total_samples"] = len(database["samples"])
    
    # Salva JSON
    with open(output_file, 'w') as f:
        json.dump(database, f, indent=2)
    
    print(f"\n✓ Database salvato: {output_file}")
    return database

def query_database(db_file="organoid_db.json", **filters):
    """Query il database con filtri"""
    with open(db_file, 'r') as f:
        db = json.load(f)
    
    results = db["samples"]
    
    # Applica filtri
    for key, value in filters.items():
        results = [s for s in results if s.get(key) == value]
    
    return results

import numpy as np

def get_n_layer_array(db_file="organoid_db.json", **filters):
    """Estrae array di tutti i N_layer dai samples filtrati"""
    results = query_database(db_file, **filters)
    
    n_layers = [sample["N_layer"] for sample in results]
    
    print(f"📊 {len(n_layers)} samples trovati")
    print(f"   Min: {min(n_layers)}, Max: {max(n_layers)}")
    print(f"   Mean: {sum(n_layers)/len(n_layers):.1f}, Std: {np.std(n_layers):.1f}")
    
    return np.array(n_layers)

# ============= USO =============
if __name__ == "__main__":
    # # 1. CREA DATABASE
    urls = ['/Volumes/Elements/Organoides/Noyaux/Chouxfleurs','/Volumes/Elements/Organoides/Noyaux/Compact','/Volumes/Elements/Organoides/Noyaux/Cystiques']
    output_files = ["organoid_chouxfleurs.json","organoid_compact.json","organoid_cystiques.json"]
    # for url, output_file in zip(urls, output_files):
    #     db = create_organoid_database(url, output_file=output_file)

    labs = ["Metatox (Paris)", "IPMC (Nice)"]
    magnifications = ["20x", "40x"]
    formats = ["uint8", "uint16"]
    resolutions = ["512x512", "1024x1024", "2048x2048"]
    
    # 2. QUERY ESEMPI
    for output_file in output_files:
        print(f"\n=== {output_file} ===")
        for lab in labs:
        #for mag in magnifications:
            #for fmt in formats:
                #for res in resolutions:
                    results = query_database(
                        db_file=output_file,
                        lab=lab,
                        #magnification=mag,
                        #format=fmt,
                        #resolution=res
                    )
                    if results:
                        #print(f"\n--- Risultati per Lab: {lab}, Mag: {mag}, Format: {fmt}, Res: {res} --- total: {len(results)}")
                        #print(f"\n--- Risultati Format: {fmt}, Res: {res} --- total: {len(results)}")
                        #print(f"\n--- Risultati Mag: {mag}, Res: {res} --- total: {len(results)}")
                        #print(f"\n--- Risultati Mag: {mag}, Format: {fmt} --- total: {len(results)}")
                        print(f"\n--- Risultati Lab: {lab} --- total: {len(results)}")

    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy import stats

    # Assumendo che tu abbia già:
    all_layers = get_n_layer_array()
    nice_layers = get_n_layer_array(lab="IPMC (Nice)")
    paris_layers = get_n_layer_array(lab="Metatox (Paris)")

    # Configurazione stile globale
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Dati (assumendo già caricati)
    all_layers = get_n_layer_array()
    nice_layers = get_n_layer_array(lab="IPMC (Nice)")
    paris_layers = get_n_layer_array(lab="Metatox (Paris)")

    # ========== PLOT 1: Istogramma Tutti i Dati ==========
    plt.figure(figsize=(10, 6))
    plt.hist(all_layers, bins=30, alpha=0.7, edgecolor='black', density=True, color='steelblue')
    plt.axvline(all_layers.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {all_layers.mean():.1f}')
    plt.axvline(np.median(all_layers), color='orange', linestyle='--', linewidth=2, 
            label=f'Median: {np.median(all_layers):.1f}')
    plt.axvline(128, color='green', linestyle=':', linewidth=2, label='Target: 128 (preprocessing)')
    plt.xlabel('Number of Z-layers')
    plt.ylabel('Density')
    plt.title(f'Z-Depth Distribution - All Samples (N={len(all_layers)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('01_zdepth_all_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ========== PLOT 2: Confronto Laboratori ==========
    plt.figure(figsize=(10, 6))
    plt.hist(nice_layers, bins=25, alpha=0.6, label=f'IPMC (Nice, N={len(nice_layers)})', 
            density=True, color='skyblue')
    plt.hist(paris_layers, bins=25, alpha=0.6, label=f'Metatox (Paris, N={len(paris_layers)})', 
            density=True, color='coral')
    plt.xlabel('Number of Z-layers')
    plt.ylabel('Density')
    plt.title('Z-Depth Distribution by Laboratory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_zdepth_labs_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ========== PLOT 3: Boxplot ==========
    plt.figure(figsize=(8, 6))
    box_data = [nice_layers, paris_layers]
    box_labels = ['IPMC (Nice)', 'Metatox (Paris)']
    bp = plt.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('coral')
    plt.ylabel('Number of Z-layers')
    plt.title('Z-Depth Distribution by Laboratory (Boxplot)')
    plt.axhline(128, color='green', linestyle=':', linewidth=2, label='Target: 128')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_zdepth_labs_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ========== PLOT 4: Tabella Statistiche ==========
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75']
    stats_data = np.array([
        [all_layers.mean(), nice_layers.mean(), paris_layers.mean()],
        [np.median(all_layers), np.median(nice_layers), np.median(paris_layers)],
        [np.std(all_layers), np.std(nice_layers), np.std(paris_layers)],
        [all_layers.min(), nice_layers.min(), paris_layers.min()],
        [all_layers.max(), nice_layers.max(), paris_layers.max()],
        [np.percentile(all_layers,25), np.percentile(nice_layers,25), np.percentile(paris_layers,25)],
        [np.percentile(all_layers,75), np.percentile(nice_layers,75), np.percentile(paris_layers,75)]
    ])

    table = ax.table(cellText=stats_data.round(1),
                    colLabels=['All', 'Nice', 'Paris'],
                    rowLabels=metrics,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for i in range(len(metrics)):
        table[(i+1, 0)].set_facecolor('#e6f3ff')  # Azzurro chiaro per row labels
    for j in range(3):  # Header
        table[(0, j)].set_facecolor('#4472c4')
        table[(0, j)].set_text_props(weight='bold', color='white')

    ax.set_title('Z-Depth Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('04_zdepth_statistics_table.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ 4 PLOT salvati:")
    print("   01_zdepth_all_histogram.pdf")
    print("   02_zdepth_labs_comparison.pdf") 
    print("   03_zdepth_labs_boxplot.pdf")
    print("   04_zdepth_statistics_table.pdf")

    def get_n_layers_by_phenotype(output_files):
        """Carica N_layers per ogni fenotipo"""
        phenotype_data = {}
        
        for file_path in output_files:
            phenotype = file_path.split('_')[1].split('.')[0].capitalize()  # chouxfleurs → Chouxfleurs
            if phenotype == "Chouxfleurs":
                phenotype = "Cauliflower"
                
            print(f"🔄 Loading {file_path}...")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            layers = [s["N_layer"] for s in data["samples"]]
            phenotype_data[phenotype] = np.array(layers)
            print(f"   → {len(layers)} samples, Mean: {np.mean(layers):.1f}")
        
        return phenotype_data

    # ========== CARICA DATI ==========
    output_files = ["organoid_chouxfleurs.json", "organoid_compact.json", "organoid_cystiques.json"]
    phenotype_layers = get_n_layers_by_phenotype(output_files)

    # ========== ISTOGRAMMA COMPARATIVO ==========
    plt.figure(figsize=(12, 8))

    colors = ['#2E8B57', '#FF8C00', '#8B0000']  # Verde, Arancione, Rosso
    labels = list(phenotype_layers.keys())
    alpha = 0.7

    for i, (phenotype, layers) in enumerate(phenotype_layers.items()):
        plt.hist(layers, bins=25, alpha=alpha, label=f'{phenotype} (N={len(layers)})', 
                density=True, color=colors[i], edgecolor='black')

    plt.xlabel('Number of Z-layers')
    plt.ylabel('Density')
    plt.title('Z-Depth Distribution by Phenotypic Class')
    plt.legend(loc='upper right')
    plt.axvline(128, color='black', linestyle=':', linewidth=2, 
            label='Preprocessing target: 128 slices')
    plt.grid(True, alpha=0.3)

    # Aggiungi statistiche nel plot
    stats_text = '\n'.join([
        f'{label}: μ={np.mean(layers):.1f}, σ={np.std(layers):.1f}' 
        for label, layers in phenotype_layers.items()
    ])
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('05_zdepth_phenotype_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Salvato: 05_zdepth_phenotype_comparison.pdf")
    print("\n📊 Riepilogo per classe:")
    for phenotype, layers in phenotype_layers.items():
        print(f"   {phenotype:12}: N={len(layers)}, Mean={np.mean(layers):.1f}, "
            f"Std={np.std(layers):.1f}, Range=[{layers.min()}-{layers.max()}]")


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
    else:
        return "40x"  # Default

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

# ============= USO =============
if __name__ == "__main__":
    # # 1. CREA DATABASE
    # url = '/Volumes/Elements/Organoides/Noyaux/Cystiques'
    output_files = ["organoid_chouxfleurs.json","organoid_compact.json","organoid_cystiques.json"]
    # db = create_organoid_database(url, output_file=output_file)
    labs = ["Metatox (Paris)", "IPMC (Nice)"]
    magnifications = ["20x", "40x"]
    formats = ["uint8", "uint16"]
    resolutions = ["512x512", "1024x1024", "2048x2048"]
    
    # 2. QUERY ESEMPI
    for output_file in output_files:
        print(f"\n=== {output_file} ===")
        for lab in labs:
            for mag in magnifications:
                for fmt in formats:
                    for res in resolutions:
                        results = query_database(
                            db_file=output_file,
                            lab=lab,
                            magnification=mag,
                            format=fmt,
                            resolution=res
                        )
                        if results:
                            print(f"\n--- Risultati per Lab: {lab}, Mag: {mag}, Format: {fmt}, Res: {res} --- total: {len(results)}")
        

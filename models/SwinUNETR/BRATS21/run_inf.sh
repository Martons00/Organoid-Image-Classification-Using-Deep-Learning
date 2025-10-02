#!/bin/bash

if [ -f swin_unetr_env/bin/activate ]; then
    source swin_unetr_env/bin/activate
else
    # Per Windows
    source swin_unetr_env/Scripts/activate
fi

echo "Environment virtuale Python attivato"


# Verifica che il file .tif esista
TIF_FILE="$1"
if [ -z "$TIF_FILE" ]; then
    echo "Uso: ./run.sh <file.tif>"
    echo "Esempio: ./run.sh brain_scan.tif"
    exit 1
fi

if [ ! -f "$TIF_FILE" ]; then
    echo "Errore: File $TIF_FILE non trovato"
    exit 1
fi

echo "Esecuzione inferenza su $TIF_FILE"


# Esegue l'inferenza
echo "Avvio inferenza..."
python inference.py "$TIF_FILE" --output_dir output

echo "Inferenza completata! Risultati salvati in ./output/"

# Mostra i risultati
if [ -d "output" ]; then
    echo "File generati:"
    ls -la output/
fi
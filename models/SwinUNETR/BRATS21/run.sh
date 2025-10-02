#!/bin/bash

# Crea ed entra nell'environment virtuale Python
python -m venv swin_unetr_env

# Attiva l'environment virtuale
if [ -f swin_unetr_env/bin/activate ]; then
    source swin_unetr_env/bin/activate
else
    # Per Windows
    source swin_unetr_env/Scripts/activate
fi

echo "Environment virtuale Python attivato"

# Aggiorna pip
pip install --upgrade pip

# Installa i requisiti 
if [ -f requirements.txt ]; then
    echo "Installazione requisiti da requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt non trovato. Installazione dipendenze base..."
    pip install torch torchvision 
    pip install monai
    pip install nibabel
    pip install numpy
    pip install matplotlib
    pip install scipy
    pip install tensorboard
    pip install tifffile
fi

echo "Dipendenze installate"

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

# Scarica modello pre-trained se non esiste
MODEL_DIR="pretrained_models"
MODEL_NAME="fold1_f48_ep300_4gpu_dice0_9059.pth"

if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
fi

if [ ! -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Download del modello pre-trained..."
    cd "$MODEL_DIR"
    wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold1_f48_ep300_4gpu_dice0_9059.zip
    unzip fold1_f48_ep300_4gpu_dice0_9059.zip
    cd ..
fi

# Esegue l'inferenza
echo "Avvio inferenza..."
python inference.py "$TIF_FILE" --output_dir output

echo "Inferenza completata! Risultati salvati in ./output/"

# Mostra i risultati
if [ -d "output" ]; then
    echo "File generati:"
    ls -la output/
fi
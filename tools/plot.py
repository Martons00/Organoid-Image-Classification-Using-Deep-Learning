import re
import matplotlib.pyplot as plt
import numpy as np

def extract_avg_losses(file_path):
    """
    Estrae tutti i valori avg_loss da un file .out di training.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regex per catturare avg_loss=numero (es. avg_loss=1.0942)
    pattern = r'avg_loss=([\d.]+)'
    matches = re.findall(pattern, content)
    
    return [float(loss) for loss in matches]

# USO: sostituisci 'tuo_file.out' con il path del tuo file
file_path = './outputs/OrganoidsINRIA_reduced/densenet/07_64/OAR_2135702.out'  # <-- CAMBIA QUI
losses = extract_avg_losses(file_path)

print(f"Array avg_loss estratti: {losses}")
print(f"Numero di epochs: {len(losses)}")

# Plot dei valori
plt.figure(figsize=(12, 6))
epochs = np.arange(1, len(losses) + 1)
plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=4, label='Avg Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Average Loss per Epoch')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

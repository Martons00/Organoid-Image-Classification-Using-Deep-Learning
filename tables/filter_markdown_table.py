import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def parse_markdown_table(content: str) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """
    Parsa una tabella Markdown e ritorna l'intestazione, i nomi delle colonne e i dati.

    Args:
        content: Contenuto del file markdown

    Returns:
        Tupla di (titolo_tabella, nomi_colonne, lista_righe)
    """
    lines = content.split('\n')

    # Trova il titolo (riga con ##)
    title = ""
    table_start = 0
    for i, line in enumerate(lines):
        if line.startswith('##'):
            title = line
            table_start = i + 1
            break

    # Trova l'inizio della tabella (riga con |)
    header_line = ""
    separator_line = ""
    data_start = 0

    for i in range(table_start, len(lines)):
        if lines[i].strip().startswith('|'):
            if header_line == "":
                header_line = lines[i]
            elif separator_line == "":
                separator_line = lines[i]
                data_start = i + 1
                break

    # Estrai i nomi delle colonne
    header_parts = [cell.strip() for cell in header_line.split('|')[1:-1]]

    # Estrai i dati
    rows = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line.startswith('|') or line == "":
            continue

        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if len(cells) == len(header_parts):
            row_dict = {header_parts[j]: cells[j] for j in range(len(header_parts))}
            rows.append(row_dict)

    return title, header_parts, rows


def filter_table(title: str, columns: List[str], rows: List[Dict[str, str]], 
                selected_columns: List[str], selected_rows: List[int] = None) -> str:
    """
    Filtra la tabella per colonne e righe selezionate.

    Args:
        title: Titolo della tabella
        columns: Lista di tutti i nomi delle colonne
        rows: Lista di tutte le righe
        selected_columns: Colonne da mantenere
        selected_rows: Indici delle righe da mantenere (None = tutte)

    Returns:
        Stringa formattata Markdown della tabella filtrata
    """
    if selected_rows is None:
        selected_rows = list(range(len(rows)))

    # Valida le colonne selezionate
    valid_columns = [col for col in selected_columns if col in columns]
    if not valid_columns:
        raise ValueError(f"Nessuna colonna valida selezionata. Colonne disponibili: {columns}")

    # Costruisci la tabella Markdown
    md_lines = [title, ""]

    # Header
    header = "| " + " | ".join(valid_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(valid_columns)) + " |"

    md_lines.append(header)
    md_lines.append(separator)

    # Righe
    for row_idx in selected_rows:
        if row_idx < len(rows):
            row = rows[row_idx]
            row_cells = [row.get(col, "") for col in valid_columns]
            row_line = "| " + " | ".join(row_cells) + " |"
            md_lines.append(row_line)

    return "\n".join(md_lines)


def process_markdown_table(input_file: str, output_file: str, 
                          selected_columns: List[str], 
                          selected_rows: List[int] = None):
    """
    Legge un file markdown, filtra la tabella e salva il risultato.

    Args:
        input_file: Percorso del file markdown di input
        output_file: Percorso del file markdown di output
        selected_columns: Colonne da mantenere
        selected_rows: Indici delle righe da mantenere (None = tutte)
    """
    # Leggi il file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parsa la tabella
    title, columns, rows = parse_markdown_table(content)

    print(f"✓ Tabella parsata")
    print(f"  - Titolo: {title}")
    print(f"  - Colonne totali: {len(columns)}")
    print(f"  - Righe totali: {len(rows)}")

    # Filtra la tabella
    filtered_md = filter_table(title, columns, rows, selected_columns, selected_rows)

    # Salva il file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(filtered_md)

    print(f"\n✓ File salvato: {output_file}")
    print(f"  - Colonne selezionate: {len(selected_columns)}")
    print(f"  - Righe selezionate: {len(selected_rows) if selected_rows else len(rows)}")

    # Restituisci a partire dalla 4ª riga (indice 3)
    lines = filtered_md.splitlines()
    result = "\n".join(lines[4:]) if len(lines) > 3 else ""
    return result


# Esempio di utilizzo
if __name__ == "__main__":

    file_names_full = [
        "H_resolution_results.md",
        "R_resolution_results.md",
    ]

    folders = [
        "densenet/",
        "resnet18/",
        "resnet50/",
        "swinunetr/",
        "swinunetr+noah/",
        "swinvit/",
    ]

    SELECTED_COLUMNS = [
    "Run",
    "Model",
    "Aug",
    "ROI",
    "PatchMerging",
    "Loss",
    "MaxEpochs",
    "LR",
    "Optim",
    "LRschedule",
    "TrainAcc",
    "ValAcc",
    "TestAcc",
    ]

    SELECTED_ROWS = None  # Tutte le righe

    row_full = []
    complete = []
    for fold in folders:
        INPUT_FILE = "results/complete/" + fold + file_names_full[0]
        OUTPUT_FOLDER = "results/light/" + fold 
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE = OUTPUT_FOLDER + file_names_full[0]

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        complete.extend(lines[3:])

        try:
            row = process_markdown_table(INPUT_FILE, OUTPUT_FILE, SELECTED_COLUMNS, SELECTED_ROWS)
            row_full.append(row[:-1])  # Rimuovi newline finale
        except FileNotFoundError:
            print(f"✗ Errore: File '{INPUT_FILE}' non trovato")
        except Exception as e:
            print(f"✗ Errore: {e}")
    
    header = "| " + " | ".join(SELECTED_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(SELECTED_COLUMNS)) + " |"
    with open("results/light/H_results.md", 'w', encoding='utf-8') as f:
        f.write("## Full Results\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for row in row_full:
            f.write(row + "\n")
    
    with open("results/complete/H_results.md", 'w', encoding='utf-8') as f:
        f.write("## Complete Results\n\n")
        f.write(lines[0] + "\n")
        f.write(lines[1] + "\n")
        f.write(lines[2] + "\n")
        for content in complete:
            f.write(content + "\n")
    
    row_reduced = []
    complete_reduced = []
    for fold in folders:
        INPUT_FILE = "results/complete/" + fold + file_names_full[1]
        OUTPUT_FOLDER = "results/light/" + fold 
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE = OUTPUT_FOLDER + file_names_full[1]

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        complete_reduced.extend(lines[3:])

        try:
            row = process_markdown_table(INPUT_FILE, OUTPUT_FILE, SELECTED_COLUMNS, SELECTED_ROWS)
            row_reduced.append(row[:-1])  # Rimuovi newline finale
        except FileNotFoundError:
            print(f"✗ Errore: File '{INPUT_FILE}' non trovato")
        except Exception as e:
            print(f"✗ Errore: {e}")
    
    header = "| " + " | ".join(SELECTED_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(SELECTED_COLUMNS)) + " |"
    with open("results/light/R_results.md", 'w', encoding='utf-8') as f:
        f.write("## Full Results\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for row in row_reduced:
            f.write(row + "\n")

    with open("results/complete/R_results.md", 'w', encoding='utf-8') as f:
        f.write("## Complete Results\n\n")
        f.write(lines[0] + "\n")
        f.write(lines[1] + "\n")
        f.write(lines[2] + "\n")
        for content in complete_reduced:
            f.write(content + "\n")


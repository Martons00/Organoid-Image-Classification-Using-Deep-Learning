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


# Esempio di utilizzo
if __name__ == "__main__":

    file_names = [
        "swinunetr_layer4",
        "swinunetr_reduced",
        "swinunetr_encoder10",
        "resnet18",
        "resnet18_reduced",
        "resnet50",
        "resnet50_reduced",
    ]


    SELECTED_COLUMNS = [
    "Run",
    "Model",
    "Aug",
    "Loss",
    "MaxEpochs",
    "LR",
    "Optim",
    "LRschedule",
    "ExactClass",
    "SplitMethod",
    "TrainAcc",
    "ValAcc",
    ]

    # Righe da mantenere (indici: 0 = prima riga dati, None = tutte le righe)
    # Esempi:
    #   None                    -> tutte le righe
    #   [0, 1, 2]             -> solo righe 0, 1, 2
    #   list(range(0, 5))      -> prime 5 righe
    #   [2, 5, 8]             -> righe specifiche
    SELECTED_ROWS = None  # Tutte le righe


    for file_name in file_names:
        INPUT_FILE = file_name + ".md"
        OUTPUT_FILE = "light/" + file_name + ".md"

        try:
            process_markdown_table(INPUT_FILE, OUTPUT_FILE, SELECTED_COLUMNS, SELECTED_ROWS)
        except FileNotFoundError:
            print(f"✗ Errore: File '{INPUT_FILE}' non trovato")
        except Exception as e:
            print(f"✗ Errore: {e}")

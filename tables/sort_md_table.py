#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from typing import List, Tuple


def split_md_row(line: str) -> List[str]:
    # Rimuove bordi '|' e spazi, poi separa per pipe
    return [c.strip() for c in line.strip().strip("|").split("|")]


def join_md_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def parse_md_table(lines: List[str]) -> Tuple[int, int, List[str], List[List[str]]]:
    """
    Ritorna (start_idx, end_idx, header_cells, data_rows)
    dove [start_idx:end_idx] sono le righe della tabella nel file.
    """
    n = len(lines)
    # Trova il blocco tabella: riga header e riga separatrice successiva
    start = -1
    for i in range(n - 1):
        if lines[i].strip().startswith("|") and lines[i+1].strip().startswith("|"):
            # Heuristic: seconda riga è separatore se contiene almeno tre '-'
            sep_candidate = lines[i+1].replace(" ", "")
            if set(sep_candidate) <= set("|-:"):
                start = i
                break
    if start == -1:
        raise RuntimeError("Tabella Markdown non trovata nel file.")

    # Estrai righe della tabella finché iniziano con '|'
    end = start + 2
    while end < n and lines[end].strip().startswith("|"):
        end += 1

    header_cells = split_md_row(lines[start])
    # salta la riga separatrice (start+1)
    data_rows = [split_md_row(l) for l in lines[start+2:end]]

    return start, end, header_cells, data_rows


def to_number(x: str):
    try:
        if x.isdigit() or (x.startswith("-") and x[1:].isdigit()):
            return int(x)
        return float(x.replace(",", "."))  # supporto decimali con virgola
    except Exception:
        return x  # lascia come stringa se non numerico


def sort_table_by_key(header: List[str], rows: List[List[str]], key: str, desc: bool) -> List[List[str]]:
    # Mappa nome colonna -> indice
    idx = {name: i for i, name in enumerate(header)}
    if key not in idx:
        raise KeyError(f"Colonna '{key}' non trovata nell'header: {header}")
    k = idx[key]

    # Ordina convertendo la chiave in numero quando possibile
    return sorted(rows, key=lambda r: (to_number(r[k]),), reverse=desc)


def rebuild_table(header: List[str], rows: List[List[str]]) -> List[str]:
    header_line = join_md_row(header)
    sep_line = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = [join_md_row(r) for r in rows]
    return [header_line, sep_line, *body_lines]


def add_rank_column(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Aggiunge 'Rank' come prima colonna e assegna rank 1..N
    in base all'ordine corrente delle righe.
    """
    new_header = ["Rank"] + header
    new_rows = []
    for i, row in enumerate(rows, start=1):
        new_rows.append([str(i)] + row)
    return new_header, new_rows


def main():
    ap = argparse.ArgumentParser(description="Ordina una tabella Markdown per colonna.")
    ap.add_argument("input", help="Percorso del file .md con la tabella")
    ap.add_argument("--key", default="ValAcc", help="Nome colonna su cui ordinare (default: ValAcc)")
    ap.add_argument("--desc", action="store_true", help="Ordina in ordine decrescente")
    ap.add_argument("--inplace", action="store_true", help="Sovrascrive il file di input")
    ap.add_argument("--output", default=None, help="File di output (se non --inplace)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    try:
        start, end, header, data_rows = parse_md_table(lines)
    except RuntimeError as e:
        print(f"Errore: {e}")
        print(f"Nessun file di nome '{args.input}' modificato.")
        return

    # Ordina
    sorted_rows = sort_table_by_key(header, data_rows, args.key, args.desc)

    # Aggiungi colonna Rank *dopo* l'ordinamento
    header_with_rank, rows_with_rank = add_rank_column(header, sorted_rows)

    # Ricostruisci tabella
    new_table_lines = rebuild_table(header_with_rank, rows_with_rank)

    # Rimpiazza nel file
    new_lines = lines[:start] + new_table_lines + lines[end:]

    if args.inplace:
        out_path = args.input
    else:
        out_path = args.output or (args.input.rsplit(".", 1)[0] + f".sorted_by_{args.key}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"Tabella ordinata per '{args.key}' ({'desc' if args.desc else 'asc'}) con colonna Rank salvata in: {out_path}")


if __name__ == "__main__":
    main()

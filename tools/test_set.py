import argparse
import math
import random
from pathlib import Path
from shutil import move

def build_test_set(src_root: Path, dest_root: Path, ratio: float, seed: int):
    random.seed(seed)
    dest_root.mkdir(parents=True, exist_ok=True)

    # Cartelle di primo livello (classi)
    class_dirs = [p for p in src_root.iterdir() if p.is_dir()]

    report = []
    for class_dir in class_dirs:
        # Raccoglie tutti i file ricorsivamente nella classe
        files = [p for p in class_dir.rglob("*") if p.is_file()]
        n_total = len(files)
        if n_total == 0:
            report.append((class_dir.name, 0, 0))
            continue

        n_take = max(1, math.ceil(n_total * ratio))
        n_take = min(n_take, n_total)
        chosen = random.sample(files, n_take)

        moved = 0
        for f in chosen:
            # Percorso relativo alla root sorgente
            rel = f.relative_to(src_root)
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Evita collisioni di nome se già esiste
            if dst.exists():
                stem, suf = dst.stem, dst.suffix
                k = 1
                new_dst = dst.with_name(f"{stem}__m{k}{suf}")
                while new_dst.exists():
                    k += 1
                    new_dst = dst.with_name(f"{stem}__m{k}{suf}")
                dst = new_dst

            move(str(f), str(dst))
            moved += 1

        report.append((class_dir.name, moved, n_total))

    print("Creato test_set (spostato) in:", dest_root)
    for name, n_take, n_total in report:
        print(f"{name}: {n_take}/{n_total} file spostati")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sposta il 10% (configurabile) dei file da ogni classe preservando la struttura."
    )
    parser.add_argument("--ratio", type=float, default=0.10, help="Frazione per cartella (es. 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la riproducibilità")

    args = parser.parse_args()
    source = "/home/mraffael/martone_project/Organoids_Dataset_256/train_set"
    dest = "/home/mraffael/martone_project/Organoids_Dataset_256/test_set"
    src_root = Path(source).resolve()
    dest_root = (src_root / dest).resolve() if not Path(dest).is_absolute() else Path(dest).resolve()
    build_test_set(src_root, dest_root, args.ratio, args.seed)

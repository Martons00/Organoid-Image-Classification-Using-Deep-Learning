#!/usr/bin/env python3
import os
import sys
import argparse

from datetime import datetime
import re
import ast
from typing import Any, Dict, Optional

HEADER = [
    "Date",
    "Run","Model","Aug","ExactClass","ROI","Loss","SimLoss","Checkpoint","Encoder10",
    "Batch","MaxEpochs","Warmup","LR","Optim","Momentum","LRschedule",
    "EarlyStop","PatchMerging","SplitMethod",
        "TrainAcc", "TrainLoss",
        "ValAcc", 
        "W_ValF1", "W_ValPrecision", "W_ValRecall", "W_Specificity", 
        "Note"
]

def yesno(v: Any) -> "str":
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return "yes" if v != 0 else "no"
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in {"true", "yes", "y", "1"}:
            return "yes"
        if lv in {"false", "no", "n", "0", ""}:
            return "no"
    return "no" if v in (None, "", 0) else "yes"

def get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default)

def fmt_roi(cfg: Dict[str, Any]) -> str:
    x, y, z = cfg.get("roi_x"), cfg.get("roi_y"), cfg.get("roi_z")
    if x and y and z:
        return f"{x}x{y}x{z}"
    return "-"

def fmt_simloss(cfg: Dict[str, Any]) -> str:
    name = (cfg.get("similarity_loss") or "").strip() if isinstance(cfg.get("similarity_loss"), str) else cfg.get("similarity_loss")
    w = cfg.get("similarity_loss_weight", 0.0)
    try:
        w = float(w)
    except Exception:
        w = 0.0
    if name and w > 0:
        return f"{name} (w={w:g})"
    return "none"

def fmt_checkpoint(cfg: Dict[str, Any]) -> str:
    ck = cfg.get("checkpoint", None)
    resume = bool(cfg.get("resume_ckpt", False))
    if isinstance(ck, str) and ck.strip():
        return ck
    if resume:
        return "resume"
    return "none"

def fmt_weight_decay(cfg: Dict[str, Any]) -> str:
    wd = cfg.get("reg_weight", None)
    if wd is None:
        wd = cfg.get("weight_decay", None)
    if wd is None:
        return "-"
    try:
        return f"{float(wd):.6g}"
    except Exception:
        return str(wd)

def canon_optim(name: Optional[str]) -> str:
    if not name:
        return "-"
    n = name.strip().lower()
    mapping = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop", "adagrad": "Adagrad"}
    return mapping.get(n, name)

def has_aug(cfg: Dict[str, Any]) -> str:
    if "augmentation" in cfg:
        return yesno(cfg.get("augmentation"))
    for k, v in cfg.items():
        if isinstance(k, str) and k.endswith("_prob") and ("Rand" in k or "rand" in k):
            try:
                if float(v) > 0:
                    return "yes"
            except Exception:
                continue
    return "no"

def parse_first_dict_from_log(log_path: str) -> Optional[Dict[str, Any]]:
    """
    Cerca il primo blocco dizionario stampato (multi-riga) e lo converte in dict Python via ast.literal_eval.
    Rileva l'inizio al primo carattere '{' dopo 'INFO' (o comunque nel testo riga) e chiude al pareggio delle parentesi.
    """
    buf = []
    brace = 0
    capturing = False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not capturing:
                if "{" in line:
                    # prendi dal primo '{' in poi
                    start = line.index("{")
                    chunk = line[start:].rstrip()
                    brace = chunk.count("{") - chunk.count("}")
                    buf.append(chunk)
                    capturing = True
                    if brace <= 0:
                        break
            else:
                chunk = line.strip()
                brace += chunk.count("{") - chunk.count("}")
                buf.append(chunk)
                if brace <= 0:
                    break
    if not buf:
        return None
    text = "\n".join(buf)
    try:
        return ast.literal_eval(text)
    except Exception as e:
        print(f"Avviso: parsing fallito per {log_path}: {e}", file=sys.stderr)
        return None

def find_training_log(exp_dir: str) -> Optional[str]:
    # Preferisci training.log nella root della cartella esperimento, altrimenti cerca il .log più recente ricorsivamente
    cand = os.path.join(exp_dir, "training.log")
    if os.path.isfile(cand):
        return cand
    latest = None
    latest_mtime = None
    for root, _, files in os.walk(exp_dir):
        for fn in files:
            if fn.endswith(".log"):
                p = os.path.join(root, fn)
                try:
                    mtime = os.path.getmtime(p)
                except Exception:
                    continue
                if latest is None or mtime > latest_mtime:
                    latest, latest_mtime = p, mtime
    return latest

def extract_date_from_log(log_path: str) -> str:
    # 1) prova a catturare il primo timestamp tipo "YYYY-MM-DD HH:MM:SS"
    ts_regex = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}:\d{2})")
    # 2) pattern nel path tipo "/YYYY-MM-DD-HH-MM/"
    path_regex = re.compile(r"(\d{4}-\d{2}-\d{2})(?:-\d{2}-\d{2})")
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ts_regex.search(line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    # fallback: cerca nel path dentro il file (es. "Log directory: .../YYYY-MM-DD-HH-MM")
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        m = path_regex.search(text)
        if m:
            return m.group(1)
    except Exception:
        pass
    # ultimo fallback: mtime del file
    try:
        return datetime.fromtimestamp(os.path.getmtime(log_path)).strftime("%Y-%m-%d")
    except Exception:
        return "-"

def collect_row(run_idx: int, exp_name: str, cfg: Dict[str, Any], date_str: str) -> Dict[str, str]:
    row = {
        "Date": date_str or "-",
        "Run": exp_name,
        "Model": str(cfg.get("model_name", "-")),
        "Aug": has_aug(cfg),
        "ExactClass": yesno(cfg.get("exact_class", False)),
        "ROI": fmt_roi(cfg),
        "Loss": str(cfg.get("loss_name", "-")),
        "SimLoss": fmt_simloss(cfg),
        "Checkpoint": "yes" if fmt_checkpoint(cfg) else "no",
        "Encoder10": "yes" if str(cfg.get("encoder10_pth", "")).strip() != "" else "no",
        "Batch": str(cfg.get("batch_size", "-")),
        "MaxEpochs": str(cfg.get("max_epochs", "-")),
        "Warmup": str(cfg.get("warmup_epochs", "-")),
        "LR": (f"{float(cfg['optim_lr']):.6g}" if cfg.get("optim_lr") is not None else "-"),
        "Optim": canon_optim(cfg.get("optim_name", "-")),
        "Momentum": (f"{float(cfg['momentum']):.6g}" if cfg.get("momentum") is not None else "-"),
        "LRschedule": str(cfg.get("lrschedule", "-")),
        "EarlyStop": yesno(cfg.get("early_stopping", False)),
        "PatchMerging": yesno(cfg.get("patch_merging", False)),
        "SplitMethod": str(cfg.get("split_method", "-")),
        "TrainAcc": "-", "TrainLoss": "-",
        "ValAcc": "-", "W_ValF1": "-", "W_ValPrecision": "-", "W_ValRecall": "-", "W_Specificity": "-", 
        "Note": "...",
    }
    return row

def to_markdown(rows, header):
    line = "| " + " | ".join(header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    lines = [line, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "-")) for h in header) + " |")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Crea tabella Markdown leggendo i config da training.log.")
    ap.add_argument("root", help="Cartella radice contenente sottocartelle degli esperimenti")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"Errore: {args.root} non è una cartella valida.", file=sys.stderr)
        sys.exit(2)

    rows = []
    run_idx = 1
    for name in sorted(os.listdir(args.root)):
        exp_dir = os.path.join(args.root, name)
        if not os.path.isdir(exp_dir):
            continue
        log_path = find_training_log(exp_dir)
        if not log_path or not os.path.isfile(log_path):
            # nessun log trovato, salta
            continue
        cfg = parse_first_dict_from_log(log_path)
        if not isinstance(cfg, dict):
            continue
        date_str = extract_date_from_log(log_path)
        rows.append(collect_row(run_idx, name, cfg, date_str))
        run_idx += 1
    print("## Experiment Results", args.root)
    print(to_markdown(rows, HEADER))
    print(f"\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Any, Dict

def yesno(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in {"true", "yes", "y", "1"}:
            return "yes"
        if lv in {"false", "no", "n", "0", ""}:
            return "no"
    return "no" if v in (None, "", 0) else "yes"

def get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_config(p):
    try:
        import yaml
    except ImportError:
        print("Errore: installa pyyaml (pip install pyyaml) per leggere config YAML.", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fmt_roi(cfg):
    x = get(cfg, "DATASET.roi_x")
    y = get(cfg, "DATASET.roi_y")
    z = get(cfg, "DATASET.roi_z")
    if x and y and z:
        return f"{x}x{y}x{z}"
    return "-"

def fmt_simloss(cfg):
    name = (get(cfg, "LOSS.similarity_loss") or "").strip()
    w = get(cfg, "LOSS.similarity_loss_weight", 0.0)
    try:
        w = float(w)
    except Exception:
        w = 0.0
    if name and w > 0:
        return f"{name} (w={w:g})"
    return "none"

def fmt_checkpoint(cfg):
    ck = (get(cfg, "MODEL.checkpoint") or "").strip()
    resume = bool(get(cfg, "MODEL.resume_ckpt", False))
    if ck:
        return ck
    if resume:
        return "resume"
    return "none"

def fmt_weight_decay(cfg):
    # Support both reg_weight (seen in sample) and weight_decay keys
    wd = get(cfg, "TRAINING.reg_weight", None)
    if wd is None:
        wd = get(cfg, "TRAINING.weight_decay", None)
    if wd is None:
        return "-"
    try:
        return f"{float(wd):.6g}"
    except Exception:
        return str(wd)

def collect_row(run_idx, exp_name, cfg):
    row = {
        "Run": str(run_idx),
        "Model": (get(cfg, "MODEL.name") or "-"),
        "Aug": yesno(get(cfg, "AUGMENTATION.augmentation", False)),
        "ExactClass": yesno(get(cfg, "DATASET.exact_class", False)),
        "ROI": fmt_roi(cfg),
        "Loss": (get(cfg, "LOSS.loss_name") or "-"),
        "SimLoss": fmt_simloss(cfg),
        "Checkpoint": fmt_checkpoint(cfg),
        "Encoder10": "yes" if (get(cfg, "MODEL.encoder10_pth") or "").strip() else "no",
        "Batch": str(get(cfg, "TRAINING.batch_size") or "-"),
        "MaxEpochs": str(get(cfg, "TRAINING.max_epochs") or "-"),
        "Warmup": str(get(cfg, "TRAINING.warmup_epochs") or "-"),
        "LR": (f"{float(get(cfg, 'TRAINING.optim_lr')):.6g}" if get(cfg, "TRAINING.optim_lr") is not None else "-"),
        "Optim": (get(cfg, "TRAINING.optim_name") or "-"),
        "Momentum": (f"{float(get(cfg, 'TRAINING.momentum')):.6g}" if get(cfg, "TRAINING.momentum") is not None else "-"),
        "LRschedule": (get(cfg, "TRAINING.lrschedule") or "-"),
        "EarlyStop": yesno(get(cfg, "TRAINING.early_stopping", False)),
        "PatchMerging": yesno(get(cfg, "TRAINING.patch_merging", False)),
        "SplitMethod": (get(cfg, "TRAINING.split_method") or "-"), 
        "TrainAcc": "-", "TrainLoss": "-",
        "ValAcc": "-", "ValF1": "-", "ValPrecision": "-", "ValRecall": "-", "Specificity": "-", 
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
    ap = argparse.ArgumentParser(description="Crea tabella Markdown dagli esperimenti (config.txt YAML).")
    ap.add_argument("root", help="Cartella radice contenente sottocartelle degli esperimenti")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"Errore: {args.root} non è una cartella valida.", file=sys.stderr)
        sys.exit(2)

    header = [
    "Run","Model","Aug","ExactClass","ROI","Loss","SimLoss","Checkpoint","Encoder10",
    "Batch","MaxEpochs","Warmup","LR","Optim","Momentum","LRschedule",
    "EarlyStop","PatchMerging","SplitMethod",
        "TrainAcc", "TrainLoss",
        "ValAcc", "ValF1", "ValPrecision", "ValRecall", "Specificity", "Note"
    ]

    rows = []
    exp_dirs = [d for d in sorted(os.listdir(args.root)) if os.path.isdir(os.path.join(args.root, d))]
    run_idx = 1
    for d in exp_dirs:
        cfg_path = os.path.join(args.root, d, "config.txt")
        if not os.path.isfile(cfg_path):
            # salta cartelle senza config.txt
            continue
        try:
            cfg = load_config(cfg_path) or {}
        except Exception as e:
            print(f"Avviso: impossibile leggere {cfg_path}: {e}", file=sys.stderr)
            continue
        rows.append(collect_row(run_idx, d, cfg))
        run_idx += 1

    print(to_markdown(rows, header))

if __name__ == "__main__":
    main()

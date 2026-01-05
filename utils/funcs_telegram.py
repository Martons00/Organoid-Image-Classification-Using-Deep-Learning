
from typing import Optional
from telegram.ext import ApplicationBuilder
from telegram import InputFile
from telegram.constants import ParseMode
from datetime import datetime
import asyncio
import os

async def send_alert(oar_id: int, message: str, token_file: str, image_path: Optional[str] = None):
    """
    Invia un messaggio di testo e un file PNG su Telegram.

    Args:
        message: testo da inviare.
        token_file: percorso del file contenente prima il token e poi la chat_id su due righe.
        image_path: percorso del file .png da inviare (opzionale).
    """
    # Leggi token e chat_id dal file
    with open(token_file, "r") as f:
        token = f.readline().strip()
        chat_id = f.readline().strip()

    # Crea l'applicazione del bot
    application = ApplicationBuilder().token(token).build()

    # Invia il messaggio testuale
    message = f"🆔 *OAR ID:* {oar_id}\n{message}"
    await application.bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.MARKDOWN)


    # Invia l'immagine PNG se fornita e il file esiste
    if image_path is not None and os.path.isfile(image_path):
        with open(image_path, "rb") as img:
            png_file = InputFile(img, filename=os.path.basename(image_path))
            await application.bot.send_photo(chat_id=chat_id, photo=png_file)

def build_training_message(args):
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Riepilogo compatto dei parametri principali
    desc = (
        f"Model: *{args.model_name}* \nDataset: *{args.dataset_name}*\nSplit Method: *{args.split_method}* \n"
        f"Epochs: *{args.max_epochs}* \nBatch: *{args.batch_size}* \nLR: *{args.optim_lr}*\n"
        + (f"Loss: *{args.loss_name}* \n" if args.loss_name else "")
        + (f"Similarity Loss: *{args.similarity_loss}* \n" if args.similarity_loss else "")
        + (f"Folds: *{args.folds}* | K: *{args.k_folds}*\n" if args.folds else "")
        + (f"DEBUG TRN: *{args.debug_train_samples} training samples*\n" if args.debug else "")
        + (f"DEBUG VAL: *{args.debug_val_samples} val samples*\n" if args.debug else "")
        + f"ROI: *{args.roi_x}x{args.roi_y}x{args.roi_z}*\n"
        + f"Optim: *{args.optim_name}* \nSched: *{args.lrschedule}*"
    )

    header = "🔔 *TRAINING START*"
    footer = f"⏱️ Start: *{time_str}* \nGPU: *{args.gpu}* | Workers: *{args.workers}*"
    bar = "─" * 10

    message = f"{header}\n{footer}\n{bar}\n{desc}\n{bar}"
    return message

def _send_telegram_safe(args, message):
    """Helper per inviare messaggi Telegram con gestione errori."""
    try:
        asyncio.run(send_alert(args.oar_id, message, token_file=args.token))
    except Exception as e:
        print(f"[Warning] Telegram notification failed: {e}")

def _send_telegram_plots(args, plots_dir, cm_dir):
    """Helper per inviare plot via Telegram."""
    plots_to_send = [
        ("*Loss Curve*", os.path.join(plots_dir, "training_loss_curve.png"), None),
        ("*Accuracy Curve*", os.path.join(plots_dir, "validation_accuracy_curve.png"), None),
        ("*Learning Rate Curve*", os.path.join(plots_dir, "learning_rate_curve.png"), None),
        ("*Loss vs LR Curve*", os.path.join(plots_dir, "loss_vs_lr_curve.png"), None),
        ("*Confusion Matrix*", os.path.join(cm_dir, "final_confusion_matrix.png"), None),
        ("*Metrics Table*", os.path.join(plots_dir, "final_metrics_table.png"), None),
    ]
    
    for msg, img_path, text_suffix in plots_to_send:
        full_msg = f"{msg}\n{text_suffix}" if text_suffix else msg
        try:
            if img_path:
                asyncio.run(send_alert(args.oar_id, full_msg, token_file=args.token, image_path=img_path))
            else:
                asyncio.run(send_alert(args.oar_id, full_msg, token_file=args.token))
        except Exception as e:
            print(f"[Warning] Failed to send telegram plot {msg}: {e}")

def _send_telegram_plots_testing(args, plots_dir, cm_dir):
    """Helper per inviare plot via Telegram."""
    plots_to_send = [
        ("*Confusion Matrix*", os.path.join(cm_dir, "final_confusion_matrix.png"), None),
        ("*Metrics Table*", os.path.join(plots_dir, "final_metrics_table.png"), None),
    ]
    
    for msg, img_path, text_suffix in plots_to_send:
        full_msg = f"{msg}\n{text_suffix}" if text_suffix else msg
        try:
            if img_path:
                asyncio.run(send_alert(args.oar_id, full_msg, token_file=args.token, image_path=img_path))
            else:
                asyncio.run(send_alert(args.oar_id, full_msg, token_file=args.token))
        except Exception as e:
            print(f"[Warning] Failed to send telegram plot {msg}: {e}")
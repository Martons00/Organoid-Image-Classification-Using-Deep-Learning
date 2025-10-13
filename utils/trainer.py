# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import shutil
import time
from tracemalloc import start

import numpy as np
import torch
from .utils import AverageMeter, distributed_all_gather
from .utils import extract_patches_5d_torch, ensure_single_channel, tile_feature_patches, plot_training_curve,plot_multi_class_training_curve,plot_loss_lr
from .data_utils import send_alert
from optimizers.early_stop import EarlyStopping  # Uncomment if used


def freeze_backbone_and_select_head_fixed(model):
    """Freezing corretto - chiama SOLO UNA VOLTA all'inizio del training"""
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'global_pool' in name or 'fc' in name or 'head' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"✓ Unfrozen: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    #print(f"Total frozen: {frozen_params}, trainable: {trainable_params}")
    return model

def train_epoch(model, loader, optimizer, epoch, loss_func, args):
    """
    Training che usa la pipeline di inferenza a patch con forward_features,
    fine-tunando solo global_pool e fc (backbone congelato).
    """
    model.train()
    losses = []

    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    start_time = time.time()
    run_loss = AverageMeter()
    total_losses = []

    for idx, batch_data in enumerate(loader):
        # Estrai data/target come nel codice esistente
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Costruisci logits per l'intero batch iterando i volumi
        batch_logits = []
        B = data.shape[0]
        for b in range(B):
            vol = data[b:b+1]                  # [1,C,D,H,W] o [1,D,H,W]
            vol = ensure_single_channel(vol, mode="first")  # -> [1,1,D,H,W]
            patches, coords = extract_patches_5d_torch(
                vol, patch_size=(args.roi_z,args.roi_y,args.roi_x), step=(args.roi_z,args.roi_y,args.roi_x), pad_value=0
            )  # patches: [N,1,128,128,128]

            # Inferenza per patch con forward_features
            feat_list = []
            optimizer.zero_grad()
            for i in range(patches.shape[0]):
                patch = patches[i:i+1].to(device).to(torch.float32)  # [1,1,128,128,128]
                feats = model.forward_features(patch)                # es. [1,Cf] o [1,Cf,1,1,N]
                feat_list.append(feats)

            feats_cat = torch.cat(feat_list, dim=0)                  # [N,Cf,...]
            feats_tiled = tile_feature_patches(feats_cat, coords=coords)  # ricostruzione feature per volume

            # Testa di classificazione: global_pool → flatten → fc
            pooled = model.global_pool(feats_tiled)                  # [1,C]
            pooled = pooled.flatten(1)                               # [1,C]
            logits_b = model.fc(pooled)                              # [1,num_classes]
            batch_logits.append(logits_b)

        logits = torch.cat(batch_logits, dim=0)                      # [B,num_classes]
        predictions = torch.softmax(logits, dim=1).argmax(dim=1)  # [B]
        #print(f"Prediction: {predictions[0]} - Target: {target[0]}")  # DEBUG
        loss = loss_func(logits, target)
        total_losses.append(loss.item())



        loss.backward()

        '''

        # Verifica che ci siano parametri con requires_grad=True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        # Debug gradienti
        grad_norm = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"Gradient norm: {grad_norm}")
        '''

        optimizer.step()

        # Aggiorna metriche come nel codice originale
        if args.distributed:
            loss_list = distributed_all_gather(
                [loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        if args.rank == 0:
            print(
                "Epoch: {}/{} Iter: {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()

    # Pulisci eventuali gradienti residui
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    return run_loss.avg

import time
import torch
import numpy as np
from torch.utils.data import DataLoader

def val_epoch(
    model,
    loader: DataLoader,
    epoch: int,
    acc_func,           # opzionale: se None, usa accuracy semplice
    args,               # deve contenere: rank, max_epochs, distributed, roi_x/y/z
):
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    start_time = time.time()
    run_acc = AverageMeter()

    # Contatori per classe
    num_classes = None
    per_class_correct = None
    per_class_total = None

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data/target
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)  # [B] o [B,1]

            # Forward a patch per volume
            batch_logits = []
            B = data.shape[0]
            for b in range(B):
                vol = data[b:b+1]                                 # [1,C,D,H,W] o [1,D,H,W]
                vol = ensure_single_channel(vol, mode="first")    # -> [1,1,D,H,W]

                patches, coords = extract_patches_5d_torch(
                    vol,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=(args.roi_z, args.roi_y, args.roi_x),
                    pad_value=0
                )  # [N,1,roi_z,roi_y,roi_x]

                feat_list = []
                for i in range(patches.shape[0]):
                    patch = patches[i:i+1].to(device).to(torch.float32)
                    feats = model.forward_features(patch)
                    feat_list.append(feats)

                feats_cat = torch.cat(feat_list, dim=0)
                feats_tiled = tile_feature_patches(feats_cat, coords=coords)  # [1,Cf,D,H,W]

                pooled = model.global_pool(feats_tiled)  # [1,Cf,1,1,1]
                pooled = pooled.flatten(1)               # [1,Cf]
                logits_b = model.fc(pooled)              # [1,num_classes]
                batch_logits.append(logits_b)

            logits = torch.cat(batch_logits, dim=0)      # [B,num_classes]

            # Inizializza i contatori per classe alla prima iterazione
            if num_classes is None:
                num_classes = logits.shape[1]
                per_class_correct = np.zeros(num_classes, dtype=np.int64)
                per_class_total = np.zeros(num_classes, dtype=np.int64)

            # Predizioni e target flat
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)                  # [B]
            target_eval = target.view(-1) if (target.ndim > 1 and target.size(-1) == 1) else target

            # Accuracy batch
            correct = (preds == target_eval).sum().item()
            not_nans = int(target_eval.numel())
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)

            # Per-class (batch) su CPU per semplicità
            t_cpu = target_eval.detach().to('cpu')
            p_cpu = preds.detach().to('cpu')
            # totali per classe
            batch_total = np.bincount(t_cpu.numpy(), minlength=num_classes)
            # corretti per classe
            mask = (p_cpu == t_cpu).numpy()
            batch_correct = np.bincount(t_cpu.numpy()[mask], minlength=num_classes)

            if getattr(args, "distributed", False):
                # All-gather dei vettori per classe
                correct_vec = torch.tensor(batch_correct, device=device, dtype=torch.float32)
                total_vec = torch.tensor(batch_total, device=device, dtype=torch.float32)
                corr_list, tot_list = distributed_all_gather(
                    [correct_vec, total_vec],
                    out_numpy=True,
                    is_valid=(idx < loader.sampler.valid_length) if hasattr(loader.sampler, "valid_length") else True
                )
                # Somma su tutti i rank
                per_class_correct += np.sum(np.stack(corr_list, axis=0), axis=0).astype(np.int64)
                per_class_total   += np.sum(np.stack(tot_list, axis=0), axis=0).astype(np.int64)

                # Aggregazione globale dell’accuracy media pesata
                acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
                n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
                acc_list, not_nans_list = distributed_all_gather(
                    [acc_tensor, n_tensor],
                    out_numpy=True,
                    is_valid=(idx < loader.sampler.valid_length) if hasattr(loader.sampler, "valid_length") else True
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(float(al), n=int(nl))
            else:
                # Single process
                per_class_correct += batch_correct.astype(np.int64)
                per_class_total   += batch_total.astype(np.int64)
                run_acc.update(acc, n=not_nans)

            if getattr(args, "rank", 0) == 0:
                print(
                    f"Val {epoch}/{args.max_epochs} {idx}/{len(loader)}, "
                    f"Acc: {run_acc.avg:.4f}, time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()

    # Calcolo accuracy per classe e packaging del risultato
    if num_classes is None:
        # Nessun batch processato
        return float('nan'), {}

    per_class_acc = {
        int(c): (float(per_class_correct[c]) / max(1, int(per_class_total[c])))
        for c in range(num_classes)
    }

    # Facoltativo: stampa riassunto
    if getattr(args, "rank", 0) == 0:
        summary = ", ".join([f"c{c}: {per_class_acc[c]:.3f}" for c in range(num_classes)])
        print(f"[Val epoch {epoch}] avg_acc={run_acc.avg:.4f} | per-class [{summary}]")

    return float(run_acc.avg), per_class_acc



def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    scheduler=None,
    start_epoch=0,
    writer_dict=None,
    final_output_dir=None,
    logger=None,
):
    if logger is not None:
        logging = logger
    writer = writer_dict["writer"] if writer_dict is not None else None
    training_losses = []
    validation_accuracies = []
    validation_per_class_accuracies = []
    lr_history = []


    val_acc_max = 0.0

    # Chiama SOLO una volta prima del training loop
    model = freeze_backbone_and_select_head_fixed(model)

    if args.early_stopping:
        early_stopping_val = EarlyStopping(mode='max', patience=args.patience_val, min_delta=args.min_delta_val, restore_best=False, verbose=True)
        early_stopping_loss = EarlyStopping(mode='min', patience=args.patience_loss, min_delta=args.min_delta_loss, restore_best=False, verbose=True)

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        logging.info(f"{args.rank} {time.ctime()} Epoch: {epoch}")
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, args=args
        )
        training_losses.append(train_loss)
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
                "lr: {:.6f}".format(optimizer.param_groups[0]["lr"]),
            )
            logging.info(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1)
                + "loss: {:.4f}".format(train_loss)
                + "time {:.2f}s".format(time.time() - epoch_time)
                + "lr: {:.6f}".format(optimizer.param_groups[0]["lr"])
            )
            lr_history.append(optimizer.param_groups[0]["lr"])
            if args.early_stopping:
                # Early Stopping step
                if early_stopping_loss.step(train_loss, model):
                    print("[EarlyStopping] stopping training for loss")
                    logging.info("[EarlyStopping] stopping training for loss")
                    if args.telegram_log:
                        message = f"*🛑 Early Stopping (Loss) Triggered at Epoch {epoch}*\n"
                        asyncio.run(send_alert(message,token_file=args.token))
                    break
            if epoch%10 == 0:
                if args.telegram_log:
                    message = f"*🏋 Final Training - Epoch {epoch}/{args.max_epochs - 1}*\nTrain Loss: {train_loss:.4f}\nBest Val Acc: {val_acc_max:.4f}\nLR: {optimizer.param_groups[0]['lr']:.6f}"
                    asyncio.run(send_alert(message,token_file=args.token))
            logging.info("" + "-" * 50)
            logging.info("")
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc,val_per_class = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
            )
            validation_accuracies.append(val_acc)
            validation_per_class_accuracies.append(val_per_class)
            if args.rank == 0:
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Val_acc:",
                    val_acc,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                logging.info(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1) +
                    ", Val_acc: {:.6f}".format(val_acc) +
                    ", time {:.2f}s".format(time.time() - epoch_time)
                )
                if args.telegram_log:
                    message = f"*✅ Final Validation - Epoch {epoch}/{args.max_epochs - 1}*\nValidation Accuracy: {val_acc:.4f}\nBest Val Acc: {val_acc_max:.4f}"
                    asyncio.run(send_alert(message,token_file=args.token))



                if writer is not None:
                    writer.add_scalar("Mean_Val", val_acc, epoch)  # Val_acc is already a float, no need to use np.mean

                val_avg_acc = val_acc  # val_acc è già il valore medio
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    logging.info("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True

                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    logging.info("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

            if args.early_stopping:
                # Early Stopping step
                if early_stopping_val.step(val_acc, model):
                    print("[EarlyStopping] stopping training for validation accuracy")
                    logging.info("[EarlyStopping] stopping training for validation accuracy")
                    if args.telegram_log:
                        message = f"*🛑 Early Stopping (Validation) Triggered at Epoch {epoch}*\n"
                        asyncio.run(send_alert(message,token_file=args.token))
                    break
            logging.info("" + "-" * 50)
            logging.info("")
            print("")

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    logging.info(f"Training Finished !, Best Accuracy: {val_acc_max}")
    if args.telegram_log:
        time_str = time.strftime('%Y/%m/%d %H-%M')
        message = f"*🏆 Training Finished!*\n{time_str}\nBest Validation Accuracy: {val_acc_max:.4f}"
        asyncio.run(send_alert(message,token_file=args.token))
    logging.info("" + "=" * 100)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    name_file = '{}_{}'.format(args.logdir, time_str)
    if final_output_dir == None:
        final_log_file = os.path.join(args.output_dir, name_file)
        plot_training_curve(training_losses, metric_name="Loss", title="Training Curve - Loss", save_path=os.path.join(final_log_file, "training_loss_curve.png"))
        plot_training_curve(lr_history, metric_name="Learning Rate", title="Training Curve - Learning Rate", save_path=os.path.join(final_log_file, "learning_rate_curve.png"))
        plot_loss_lr(training_losses, lr_history, title="Training Curve - Loss vs Learning Rate", save_path=os.path.join(final_log_file, "loss_vs_lr_curve.png"))
        plot_multi_class_training_curve(validation_accuracies, validation_per_class_accuracies, title="Training Curve - Accuracy", save_path=os.path.join(final_log_file, "validation_accuracy_curve.png"))
        if args.telegram_log:
            message = f"*📈 Training curves saved*\n{final_log_file}"
            asyncio.run(send_alert(message,token_file=args.token))
            message = f"*Loss Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_log_file, "training_loss_curve.png")))
            message = f"*Accuracy Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_log_file, "validation_accuracy_curve.png")))
            message = f"*Learning Rate Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_log_file, "learning_rate_curve.png")))
            message = f"*Loss vs Learning Rate Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_log_file, "loss_vs_lr_curve.png")))

    else:
        plot_training_curve(training_losses, metric_name="Loss", title="Training Curve - Loss", save_path=os.path.join(final_output_dir, "training_loss_curve.png"))
        plot_training_curve(lr_history, metric_name="Learning Rate", title="Training Curve - Learning Rate", save_path=os.path.join(final_output_dir, "learning_rate_curve.png"))
        plot_loss_lr(training_losses, lr_history, title="Training Curve - Loss vs Learning Rate", save_path=os.path.join(final_output_dir, "loss_vs_lr_curve.png"))
        plot_multi_class_training_curve(validation_accuracies, validation_per_class_accuracies, title="Training Curve - Accuracy", save_path=os.path.join(final_output_dir, "validation_accuracy_curve.png"))
        if args.telegram_log:
            message = f"*📈 Training curves saved*\n{final_output_dir}"
            asyncio.run(send_alert(message,token_file=args.token))
            message = f"*Loss Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_output_dir, "training_loss_curve.png")))
            message = f"*Accuracy Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_output_dir, "validation_accuracy_curve.png")))
            message = f"*Learning Rate Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_output_dir, "learning_rate_curve.png")))
            message = f"*Loss vs Learning Rate Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_output_dir, "loss_vs_lr_curve.png")))

    return val_acc_max


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
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from .utils import AverageMeter, distributed_all_gather
from .utils import extract_patches_5d_torch, ensure_single_channel, tile_feature_patches, plot_training_curve
from .data_utils import send_alert

def freeze_backbone_and_select_head(model):
    # Congela tutti i parametri
    for p in model.parameters():
        p.requires_grad = False
    # Scongela solo i layer di testa
    for p in model.global_pool.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    return model

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    """
    Training che usa la pipeline di inferenza a patch con forward_features,
    fine-tunando solo global_pool e fc (backbone congelato).
    """
    model.train()
    losses = []
    freeze_backbone_and_select_head(model)
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):
        # Estrai data/target come nel codice esistente
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Azzera gradienti solo dei layer sbloccati
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad = None

        with autocast(enabled=args.amp):
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
            print(f"Prediction: {predictions[0]} - Target: {target[0]}")  # DEBUG
            loss = loss_func(logits, target)

        # Backward/step solo sui parametri sbloccati
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,          # opzionale: se None, useremo accuracy semplice
    args,    # opzionale
):
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data/target come nel codice esistente
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)  # atteso [B] o [B,1]

            with autocast(enabled=args.amp):
                batch_logits = []
                B = data.shape[0]
                for b in range(B):
                    vol = data[b:b+1]                                   # [1,C,D,H,W] o [1,D,H,W]
                    vol = ensure_single_channel(vol, mode="first")       # -> [1,1,D,H,W]
                    patches, coords = extract_patches_5d_torch(
                    vol, patch_size=(args.roi_z,args.roi_y,args.roi_x), step=(args.roi_z,args.roi_y,args.roi_x), pad_value=0
                    )

                    feat_list = []
                    for i in range(patches.shape[0]):
                        patch = patches[i:i+1].to(device).to(torch.float32)   # [1,1,128,128,128]
                        feats = model.forward_features(patch)                  # feature per patch
                        feat_list.append(feats)

                    feats_cat = torch.cat(feat_list, dim=0)                    # [N,Cf,...]
                    feats_tiled = tile_feature_patches(feats_cat, coords=coords)  # ricostruzione feature
                    pooled = model.global_pool(feats_tiled)                    # [1,C]
                    pooled = pooled.flatten(1)                                 # [1,C]
                    logits_b = model.fc(pooled)                                # [1,num_classes]
                    batch_logits.append(logits_b)

                logits = torch.cat(batch_logits, dim=0)                        # [B,num_classes]

                # Calcolo metrica di accuratezza per classificazione
                # Se sono passati post_* (MONAI), usiamo softmax+argmax come default
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)                                    # [B]
                if target.ndim > 1 and target.size(-1) == 1:
                    target_eval = target.view(-1)
                else:
                    target_eval = target

                correct = (preds == target_eval).sum().item()
                not_nans = target_eval.numel()
                acc = correct / max(1, not_nans)

            # Aggregazione distribuita come nel codice originale
            if args.distributed:
                acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
                n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
                acc_list, not_nans_list = distributed_all_gather(
                    [acc_tensor, n_tensor],
                    out_numpy=True,
                    is_valid=idx < loader.sampler.valid_length
                )
                # media pesata per numero di esempi
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc, n=not_nans)

            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Acc:",
                    run_acc.avg,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    return run_acc.avg


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

    scaler = None
    if args.amp:
        scaler = GradScaler()

    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        logging.info(f"{args.rank} {time.ctime()} Epoch: {epoch}")
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        training_losses.append(train_loss)
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            logging.info(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1)
                + "loss: {:.4f}".format(train_loss)
                + "time {:.2f}s".format(time.time() - epoch_time)
            )
            if epoch%10 == 0:
                if args.telegram_log:
                    message = f"*Final Training - Epoch {epoch}/{args.max_epochs - 1}*\nTrain Loss: {train_loss:.4f}\nBest Val Acc: {val_acc_max:.4f}"
                    asyncio.run(send_alert(message,token_file=args.token))
            logging.info("" + "-" * 10)
            logging.info("")
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
            )
            validation_accuracies.append(val_acc)
            print("Validation :", val_acc)
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
                    message = f"*Final Validation - Epoch {epoch}/{args.max_epochs - 1}*\nValidation Accuracy: {val_acc:.4f}\nBest Val Acc: {val_acc_max:.4f}"
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

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    logging.info(f"Training Finished !, Best Accuracy: {val_acc_max}")
    if args.telegram_log:
        message = f"*Training Finished!* \nBest Validation Accuracy: {val_acc_max:.4f}"
        asyncio.run(send_alert(message,token_file=args.token))
    logging.info("" + "=" * 100)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    name_file = '{}_{}'.format(args.logdir, time_str)
    if final_output_dir == None:
        final_log_file = os.path.join(args.output_dir, name_file)
        plot_training_curve(training_losses, metric_name="Loss", title="Curva di Training - Loss", save_path=os.path.join(final_log_file, "training_loss_curve.png"))
        plot_training_curve(validation_accuracies, metric_name="Accuracy", title="Curva di Training - Accuracy", save_path=os.path.join(final_log_file, "validation_accuracy_curve.png"))
        if args.telegram_log:
            message = f"*Training curves saved*\n{final_log_file}"
            asyncio.run(send_alert(message,token_file=args.token))
            message = f"*Loss Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_log_file, "training_loss_curve.png")))
            message = f"*Accuracy Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_log_file, "validation_accuracy_curve.png")))

    else:
        plot_training_curve(training_losses, metric_name="Loss", title="Curva di Training - Loss", save_path=os.path.join(final_output_dir, "training_loss_curve.png"))
        plot_training_curve(validation_accuracies, metric_name="Accuracy", title="Curva di Training - Accuracy", save_path=os.path.join(final_output_dir, "validation_accuracy_curve.png"))
        if args.telegram_log:
            message = f"*Training curves saved*\n{final_output_dir}"
            asyncio.run(send_alert(message,token_file=args.token))
            message = f"*Loss Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_output_dir, "training_loss_curve.png")))
            message = f"*Accuracy Curve*"
            asyncio.run(send_alert(message,token_file=args.token,image_path=os.path.join(final_output_dir, "validation_accuracy_curve.png")))

    return val_acc_max

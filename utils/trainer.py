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

import logging
import os
import pdb
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from .utils import AverageMeter, distributed_all_gather
from .test import extract_patches_5d_torch, ensure_single_channel, tile_feature_patches

from monai.data import decollate_batch

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
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()

    # Pulisci eventuali gradienti residui
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    return run_loss.avg


def train_epoch_old(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def val_epoch(
    model,
    loader,
    epoch,
    acc_func,          # opzionale: se None, useremo accuracy semplice
    args,
    post_sigmoid=None, # opzionale: se forniti, li usiamo per compatibilità
    post_pred=None     # opzionale
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
                        vol, patch_size=(128,128,128), step=(128,128,128), pad_value=0
                    )  # [N,1,128,128,128]

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


def val_epoch_old(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["vol"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Dice_TC:",
                    Dice_TC,
                    ", Dice_WT:",
                    Dice_WT,
                    ", Dice_ET:",
                    Dice_ET,
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
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
    writer_dict=None,
):
    writer = writer_dict["writer"] if writer_dict is not None else None
    if writer is None:
        if args.logdir is not None and args.rank == 0:
            writer = SummaryWriter(log_dir=args.logdir)
            if args.rank == 0:
                print("Writing Tensorboard logs to ", args.logdir)
                logging.info("Writing Tensorboard logs to ", args.logdir)

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
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:
                Dice_TC = val_acc[0]
                Dice_WT = val_acc[1]
                Dice_ET = val_acc[2]
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice_TC:",
                    Dice_TC,
                    ", Dice_WT:",
                    Dice_WT,
                    ", Dice_ET:",
                    Dice_ET,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                logging.info(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1)
                    + ", Dice_TC:"
                    + str(Dice_TC)
                    + ", Dice_WT:"
                    + str(Dice_WT)
                    + ", Dice_ET:"
                    + str(Dice_ET)
                    + ", time {:.2f}s".format(time.time() - epoch_time)
                )

                if writer is not None:
                    writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_acc)
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

    return val_acc_max

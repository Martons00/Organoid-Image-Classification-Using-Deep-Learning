def train_epoch_old(model, loader, optimizer, epoch, loss_func, acc_func, args):
    """
    Training con pipeline di inferenza a patch usando forward_features.
    Fine-tuning di global_pool e fc con backbone congelato.
    """
    model.train()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    
    # Contatori per classe
    num_classes = None
    per_class_correct = None
    per_class_total = None

    
    if args.augmentation:
        train_transform = get_train_transforms()
    else:
        train_transform = None
    
    # Liste per confusion matrix
    all_preds = []
    all_targets = []
    
    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    
    for idx, batch_data in enumerate(loader):
        # Estrai data e target
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]
        
        # ============================================
        # AUGMENTATION qui, on-the-fly
        # ============================================
        if train_transform is not None:
            # Augmenta solo il 50% dei samples nel batch
            data = selective_augmentation(
                data, 
                train_transform,
                augmentation_ratio=0.5  # ← 50% originali, 50% augmentati
            )

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Costruisci logits per l'intero batch
        batch_logits = []
        feat_list_all = []
        hidden_list_all = []
        
        B = data.shape[0]
        for b in range(B):
            vol = data[b:b+1]  # [1,C,D,H,W]
            vol = ensure_single_channel(vol, mode="first")  # [1,1,D,H,W]
            
            # Estrai patch dal volume
            patches, coords = extract_patches_5d_torch(
                vol, 
                patch_size=(args.roi_z, args.roi_y, args.roi_x), 
                step=(args.roi_z, args.roi_y, args.roi_x), 
                pad_value=0
            )
            
            # Inferenza per ogni patch
            patches = patches.to(device).to(torch.float32)  # Converti una volta sola
            
            sw_batch_size = args.sw_batch_size if hasattr(args, 'sw_batch_size') else 4
            feat_list = []
            hidden_list = []

            for i in range(0, patches.shape[0], sw_batch_size):
                end_idx = min(i + sw_batch_size, patches.shape[0])
                batch_patches = patches[i:end_idx]  # [sw_batch_size, 1, 128, 128, 128]
                
                feats, hidden = model.forward_features(batch_patches)  # Forward su batch
                
                feat_list.append(feats)    # [sw_batch_size, Cf, fD, fH, fW]
                hidden_list.append(hidden) # [sw_batch_size, Ch, hD, hH, hW]

            # Concatena tutti i batch
            feats_cat = torch.cat(feat_list, dim=0)   # [N, Cf, fD, fH, fW]
            hidden_cat = torch.cat(hidden_list, dim=0) # [N, Ch, hD, hH, hW]
            # print(f"Sample {b}: num_patches={patches.shape[0]}, feats_cat shape={feats_cat.shape}")
            
            feats_tiled = tile_feature_patches(feats_cat, coords=coords)

            # print(f"Sample {b}: feats_tiled shape={feats_tiled.shape}")
            
            hidden_tiled = tile_feature_patches(hidden_cat, coords=coords)
            
            feat_list_all.append(feats_tiled)
            hidden_list_all.append(hidden_tiled)
            
            # Classificazione: global_pool → flatten → fc
            pooled = model.global_pool(feats_tiled)
            # print(f"Sample {b}: pooled shape={pooled.shape}")
            
            if args.model_name == "swinunetr+ml_decoder":
                pooled = pooled.flatten(2)
            elif args.model_name == "swinunetr" or "resnet" in args.model_name or  "densenet" in args.model_name or "swinvit" in args.model_name:
                pooled = pooled.flatten(1)
            
            # print(f"Sample {b}: flattened pooled shape={pooled.shape}")
            logits_b = model.fc(pooled)  # [1,num_classes]
            batch_logits.append(logits_b)
        
        # Calcola similarity matrices solo per epoche selezionate
        should_compute_sim = (
            (epoch == 0 or epoch == (args.max_epochs - 1) or epoch == int(args.max_epochs * 0.5)) 
            and idx == 0 
            and args.rank == 0
        )
        
        sim = None
        if should_compute_sim or args.similarity_loss in ["contrastive", "margin"]:
            feat_concat = torch.cat(feat_list_all, dim=0)  # [B,Cf,D,H,W]
            feat_flat = feat_concat.view(feat_concat.shape[0], -1)  # [B,Cf*D*H*W]
            sim = compute_similarity_matrix(feat_flat)
            
            if should_compute_sim:
                sim_np = sim.detach().float().cpu().numpy()
                plot_similarity_heatmap_new(
                    sim_np, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_epoch{epoch+1}_iter{idx}.png")
                )
                
                hidden_concat = torch.cat(hidden_list_all, dim=0)
                hidden_flat = hidden_concat.view(hidden_concat.shape[0], -1)
                sim_hidden = compute_similarity_matrix(hidden_flat).cpu().detach().numpy()
                plot_similarity_heatmap_new(
                    sim_hidden, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_hidden_epoch{epoch+1}_iter{idx}.png")
                )
        
        # Calcola loss
        logits = torch.cat(batch_logits, dim=0)  # [B,num_classes]
        loss = loss_func(logits, target)
        
        # Inizializza contatori per classe alla prima iterazione
        if num_classes is None:
            num_classes = logits.shape[1]
            per_class_correct = np.zeros(num_classes, dtype=np.int64)
            per_class_total = np.zeros(num_classes, dtype=np.int64)
        
        # Calcola metriche di accuracy
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # [B]
            target_eval = target.view(-1) if target.ndim > 1 else target
            
            all_preds.append(preds.cpu())
            all_targets.append(target_eval.cpu())
            
            # Accuracy batch
            correct = (preds == target_eval).sum().item()
            not_nans = target_eval.numel()
            
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)
            
            # Per-class accuracy
            t_cpu = target_eval.cpu().numpy()
            p_cpu = preds.cpu().numpy()
            
            # Calcola correttezza per ogni sample
            mask = (p_cpu == t_cpu)
            
            # Conta totali e corretti per classe
            batch_total = np.bincount(t_cpu, minlength=num_classes)
            batch_correct = np.bincount(t_cpu[mask], minlength=num_classes)
            
            per_class_correct += batch_correct
            per_class_total += batch_total
        
        sim_loss_value = 0.0
        if args.similarity_loss == "contrastive" and sim is not None:
            loss_sim = supervised_contrastive_from_similarity(sim, target, temperature=0.07)
            loss = loss + args.similarity_loss_weight * loss_sim
            sim_loss_value = loss_sim.item()
        elif args.similarity_loss == "margin" and sim is not None:
            loss_sim = similarity_margin_loss(sim, target, pos_margin=0.5, neg_margin=0.0)
            loss = loss + args.similarity_loss_weight * loss_sim
            sim_loss_value = loss_sim.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Aggiorna metriche
        if is_distributed:
            loss_list = distributed_all_gather(
                [loss], 
                out_numpy=True, 
                is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size
            )
            
            # Aggregazione accuracy globale
            acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
            n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
            
            acc_list, not_nans_list = distributed_all_gather(
                [acc_tensor, n_tensor],
                out_numpy=True,
                is_valid=idx < loader.sampler.valid_length
            )
            
            for al, nl in zip(acc_list, not_nans_list):
                run_acc.update(float(al), n=int(nl))
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_acc.update(acc, n=not_nans)
        
        # Logging
        if is_main_process:
            if sim_loss_value != 0.0:
                print(
                    f"Epoch: {epoch+1}/{args.max_epochs} Iter: {idx+1}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"sim_loss: {sim_loss_value:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
            else:
                print(
                    f"Epoch: {epoch+1}/{args.max_epochs} Iter: {idx+1}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
        start_time = time.time()
    
    # Calcola accuracy per classe
    per_class_acc = {
        int(c): float(per_class_correct[c]) / max(1, int(per_class_total[c]))
        for c in range(num_classes)
    }
    
    # Confusion matrix
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    # Stampa riassunto
    if is_main_process:
        summary = ", ".join([f"c{c}: {per_class_acc[c]:.3f}" for c in range(num_classes)])
        print(f"[Train epoch {epoch+1}] avg_loss={run_loss.avg:.4f} avg_acc={run_acc.avg:.4f} | per-class [{summary}]")
    
    return run_loss.avg, float(run_acc.avg), per_class_acc, cm


def val_epoch_old(model,loader: DataLoader,epoch: int,acc_func,args,) -> tuple[float, dict, np.ndarray]:
    """
    Validazione con pipeline di inferenza a patch usando forward_features.
    
    Returns:
        tuple: (avg_accuracy, per_class_accuracy_dict, confusion_matrix)
    """
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_acc = AverageMeter()
    
    # Contatori per classe
    num_classes = None
    per_class_correct = None
    per_class_total = None
    
    # Liste per confusion matrix
    all_preds = []
    all_targets = []

    # Liste per la visualizzazione degli errori
    all_errors_paths = []

    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data e target
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]

            paths = batch_data["path"] if "path" in batch_data else None
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
             # Costruisci logits per l'intero batch
            batch_logits = []
            B = data.shape[0]
            
            for b in range(B):
                vol = data[b:b+1]  # [1,C,D,H,W]
                vol = ensure_single_channel(vol, mode="first")  # [1,1,D,H,W]
                
                # Estrai patch dal volume
                patches, coords = extract_patches_5d_torch(
                    vol, 
                    patch_size=(args.roi_z, args.roi_y, args.roi_x), 
                    step=(args.roi_z, args.roi_y, args.roi_x), 
                    pad_value=0
                )
                
                # Inferenza per ogni patch
                patches = patches.to(device).to(torch.float32)  # Converti una volta sola
                
                sw_batch_size = args.sw_batch_size if hasattr(args, 'sw_batch_size') else 4
                feat_list = []

                for i in range(0, patches.shape[0], sw_batch_size):
                    end_idx = min(i + sw_batch_size, patches.shape[0])
                    batch_patches = patches[i:end_idx]  # [sw_batch_size, 1, 128, 128, 128]
                    
                    feats, _ = model.forward_features(batch_patches)  # Forward su batch
                    
                    feat_list.append(feats)    # [sw_batch_size, Cf, fD, fH, fW]

                # Concatena tutti i batch
                feats_cat = torch.cat(feat_list, dim=0)   # [N, Cf, fD, fH, fW]
                # print(f"Sample {b}: num_patches={patches.shape[0]}, feats_cat shape={feats_cat.shape}")
                
                feats_tiled = tile_feature_patches(feats_cat, coords=coords)
                
                # Classificazione
                pooled = model.global_pool(feats_tiled)
                
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr" or "resnet" in args.model_name or  "densenet" in args.model_name or "swinvit" in args.model_name:
                    pooled = pooled.flatten(1)
                
                logits_b = model.fc(pooled)  # [1,num_classes]
                batch_logits.append(logits_b)
            
            logits = torch.cat(batch_logits, dim=0)  # [B,num_classes]
            
            # Inizializza contatori per classe alla prima iterazione
            if num_classes is None:
                num_classes = logits.shape[1]
                per_class_correct = np.zeros(num_classes, dtype=np.int64)
                per_class_total = np.zeros(num_classes, dtype=np.int64)
            
            # Predizioni
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # [B]
            target_eval = target.view(-1) if target.ndim > 1 else target
            
            all_preds.append(preds.cpu())
            all_targets.append(target_eval.cpu())
            
            # Accuracy batch
            correct = (preds == target_eval).sum().item()
            all_errors_paths.extend([(paths[i], preds[i], target_eval[i]) for i in range(len(paths)) if preds[i] != target_eval[i]])
            not_nans = target_eval.numel()
            
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)
            
            # Per-class accuracy
            t_cpu = target_eval.cpu().numpy()
            p_cpu = preds.cpu().numpy()
            
            # Calcola correttezza per ogni sample
            mask = (p_cpu == t_cpu)
            
            # Conta totali e corretti per classe in un solo passaggio
            batch_total = np.bincount(t_cpu, minlength=num_classes)
            batch_correct = np.bincount(t_cpu[mask], minlength=num_classes)
            
            if is_distributed:
                # Verifica validità sample
                is_valid = idx < loader.sampler.valid_length if hasattr(loader.sampler, "valid_length") else True
                
                # All-gather per classe
                correct_vec = torch.tensor(batch_correct, device=device, dtype=torch.float32)
                total_vec = torch.tensor(batch_total, device=device, dtype=torch.float32)
                
                corr_list, tot_list = distributed_all_gather(
                    [correct_vec, total_vec],
                    out_numpy=True,
                    is_valid=is_valid
                )
                
                per_class_correct += np.sum(np.stack(corr_list, axis=0), axis=0).astype(np.int64)
                per_class_total += np.sum(np.stack(tot_list, axis=0), axis=0).astype(np.int64)
                
                # Aggregazione accuracy globale
                acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
                n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
                
                acc_list, not_nans_list = distributed_all_gather(
                    [acc_tensor, n_tensor],
                    out_numpy=True,
                    is_valid=is_valid
                )
                
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(float(al), n=int(nl))
            else:
                per_class_correct += batch_correct
                per_class_total += batch_total
                run_acc.update(acc, n=not_nans)
            
            # Logging
            if is_main_process:
                print(
                    f"Val {epoch+1}/{args.max_epochs} {idx+1}/{len(loader)}, "
                    f"Acc: {run_acc.avg:.4f}, time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()
    
    # Gestisci caso senza batch processati
    if num_classes is None:
        return float('nan'), {}, np.array([])
    
    # Calcola accuracy per classe
    per_class_acc = {
        int(c): float(per_class_correct[c]) / max(1, int(per_class_total[c]))
        for c in range(num_classes)
    }
    
    # Confusion matrix
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    # Stampa riassunto
    if is_main_process:
        summary = ", ".join([f"c{c}: {per_class_acc[c]:.3f}" for c in range(num_classes)])
        print(f"[Val epoch {epoch+1}] avg_acc={run_acc.avg:.4f} | per-class [{summary}]")

    return float(run_acc.avg), per_class_acc, cm, all_errors_paths

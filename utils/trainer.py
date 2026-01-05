# Standard library
import os
import shutil
import time

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Local imports - Tools
from tools.plots import (
    plot_training_curve,
    plot_multi_class_training_curve,
    plot_loss_lr,
)
from tools.confusion_matrix import (
    plot_confusion_matrix,
    metrics_from_confusion_matrix,
    format_print_metrics,
    plot_metrics_table,
)


# Local imports - Other
from optimizers.early_stop import EarlyStopping

from .funcs_training import train_epoch_pm, train_epoch
from .funcs_validation import val_epoch_pm, val_epoch
from .funcs_testing import test_epoch_pm, test_epoch
from .funcs_mc_dropout import test_epoch_mc_dropout, plot_uncertainty_analysis

from .funcs_telegram import (
    send_alert,
    build_training_message,
    _send_telegram_safe,
    _send_telegram_plots,
    _send_telegram_plots_testing
)


def finetune_model(model,args):
    """Freezing corretto - chiama SOLO UNA VOLTA all'inizio del training"""
    frozen_params = 0
    trainable_params = 0
    lists_of_names = []
    if args.model_name == "resnet50":
        lists_of_names = ["layer4.2","fc","global_pool"]
    elif args.model_name == "swinunetr":
        lists_of_names = ["encoder10","fc","global_pool","swinViT.layers4.0.blocks.1.","swinViT.layers4.0.downsample.","head"]
    elif args.model_name == "swinunetr+ml_decoder":
        lists_of_names = ["fc","global_pool","head"]
    elif args.model_name == "swinunetr+noah":
        lists_of_names = ["encoder10","fc","global_pool","swinViT.layers4.0.blocks.1.","swinViT.layers4.0.downsample.","head"]
    elif args.model_name == "densenet":
        lists_of_names = ["features","fc","global_pool"]
    elif args.model_name == "swinvit":
        lists_of_names = ["layers4","fc","global_pool","head"]

    for name, param in model.named_parameters():
        if any(layer in name for layer in lists_of_names):
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"✓ Unfrozen: {name} ({param.numel()} params)")
        elif args.model_name == "resnet18":
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            print(f"✗ Frozen: {name} ({param.numel()} params)")
            frozen_params += param.numel()
    
    print(f"Total frozen: {frozen_params}, trainable: {trainable_params}")
    return model

def unfreeze_model(model,args):
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param.requires_grad = True
        trainable_params += param.numel()
        print(f"✓ Unfrozen: {name} ({param.numel()} params)")
    
    print(f"Total frozen: {frozen_params}, trainable: {trainable_params}")
    return model

def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.final_output_dir, filename)
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
)-> tuple[float,float,float, dict]:
    """
    Loop di training principale con validation, early stopping e logging.
    
    Returns:
        float: Best validation accuracy raggiunta
    """
    # Setup logging e writer
    writer = writer_dict.get("writer") if writer_dict is not None else None
    
    # Inizializza liste per metriche
    training_losses = []
    training_accuracies = []
    training_per_class_accuracies = []
    validation_accuracies = []
    validation_per_class_accuracies = []
    lr_history = []

    # Inizializza lo step
    args.step= (args.roi_z, int(args.roi_y * 2 // 3), int(args.roi_x * 2 // 3))
    
    # Setup directory output
    final_output_dir = final_output_dir + "/training"
    args.final_output_dir = final_output_dir
    if final_output_dir is None:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        name_file = f'{args.logdir}_{time_str}'
        final_output_dir = os.path.join(args.output_dir, name_file)
    
    # Crea struttura directory
    final_plots_dir = os.path.join(final_output_dir, "plots")
    sim_plots_dir = os.path.join(final_plots_dir, "similarity")
    cm_plots_dir = os.path.join(final_plots_dir, "confusion_matrix")
    metrics_plots_dir = os.path.join(final_plots_dir, "metrics_tables")
    errors_log_dir = os.path.join(final_output_dir, "errors_logs")
    
    os.makedirs(final_plots_dir, exist_ok=True)
    os.makedirs(sim_plots_dir, exist_ok=True)
    os.makedirs(cm_plots_dir, exist_ok=True)
    os.makedirs(metrics_plots_dir, exist_ok=True)
    os.makedirs(errors_log_dir, exist_ok=True)

    args.final_plots_dir = final_plots_dir
    args.sim_plots_dir = sim_plots_dir
    
    # Cache attributi comuni
    is_main_process = args.rank == 0
    should_save = is_main_process and args.final_output_dir is not None and args.save_checkpoint
    use_telegram = args.telegram_log if hasattr(args, 'telegram_log') else False
    
    val_acc_max = args.best_acc if hasattr(args, 'best_acc') else 0.0
    last_cm = None
    last_metrics = None
    best_metrics = None

    if args.pretrained_model_name is None:
        model = unfreeze_model(model,args)
    else:
        model = finetune_model(model,args)

    
    # Setup early stopping
    early_stopping_val = None
    early_stopping_loss = None
    if args.early_stopping:
        early_stopping_val = EarlyStopping(
            mode='max', 
            patience=args.patience_val, 
            min_delta=args.min_delta_val, 
            restore_best=True, 
            verbose=True
        )
        early_stopping_loss = EarlyStopping(
            mode='min', 
            patience=args.patience_loss, 
            min_delta=args.min_delta_loss, 
            restore_best=False, 
            verbose=True
        )
    
    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        
        if is_main_process:
            print(f"{args.rank} {time.ctime()} Epoch: {epoch+1}")
            if logger:
                logger.info(f"{args.rank} {time.ctime()} Epoch: {epoch+1}")
        
        # Training
        epoch_time = time.time()

        if args.patch_merging:
            train_loss, train_acc, train_per_class_acc, train_cm, train_errors_paths = train_epoch_pm(
                model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, acc_func=acc_func, args=args
            )
        else:
            train_loss, train_acc, train_per_class_acc, train_cm, train_errors_paths = train_epoch(
                model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, acc_func=acc_func, args=args
            )

        training_losses.append(train_loss)

        last_train_cm = train_cm
        train_metrics = metrics_from_confusion_matrix(train_cm)
        last_train_metrics = train_metrics
        train_metrics_str = format_print_metrics(train_metrics)

        training_accuracies.append(train_acc)
        training_per_class_accuracies.append(train_per_class_acc)

        with open(os.path.join(errors_log_dir, f"training_errors.txt"), 'a') as f:
            f.write(f"Epoch {epoch+1}:\n")
            for path, pred, target in train_errors_paths:
                f.write(f"{path}\tPred: {pred}\tTarget: {target}\n")
            f.write(f"*" + "-"*40 + "*\n")
            f.write("\n")

        # Learning rate attuale
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)
        
        if is_main_process:
            train_time = time.time() - epoch_time
            msg = (
                f"Final training {epoch+1}/{args.max_epochs}, "
                f"loss: {train_loss:.4f}, time {train_time:.2f}s, lr: {current_lr:.6f}"
                f"\n{train_metrics_str}\n"
                f"*----------------------------------------*"
            )
            print(msg)
            if logger:
                logger.info(msg)
            
            # Telegram notification ogni 10 epoche
            if use_telegram and epoch % 10 == 0:
                telegram_msg = (
                    f"*🏋 Training - Epoch {epoch+1}/{args.max_epochs}*\n"
                    f"Train Loss: {train_loss:.4f}\n"
                    f"Train Acc: {train_acc:.4f}\n"
                    f"Best Val Acc: {val_acc_max:.4f}\n"
                    f"LR: {current_lr:.6f}\n"
                )
                _send_telegram_safe(args, telegram_msg)
        
        
        # Validation
        is_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            
            epoch_time = time.time()
            if args.patch_merging:
                val_loss,val_acc, val_per_class, cm, val_errors_paths = val_epoch_pm(
                    model, val_loader, epoch=epoch, acc_func=acc_func,loss_func=loss_func, args=args
                )
            else:
                val_loss,val_acc, val_per_class, cm, val_errors_paths = val_epoch(
                    model, val_loader, epoch=epoch, acc_func=acc_func,loss_func=loss_func, args=args
                )

            with open(os.path.join(errors_log_dir, f"validation_errors.txt"), 'a') as f:
                f.write(f"Epoch {epoch+1}:\n")
                for path, pred, target in val_errors_paths:
                    f.write(f"{path}\tPred: {pred}\tTarget: {target}\n")
                f.write(f"*" + "-"*40 + "*\n")
                f.write("\n")

            last_cm = cm
            metrics = metrics_from_confusion_matrix(cm)
            last_metrics = metrics
            metrics_str = format_print_metrics(metrics)
            
            validation_accuracies.append(val_acc)
            validation_per_class_accuracies.append(val_per_class)
            
            if is_main_process:
                val_time = time.time() - epoch_time
                msg = (
                    f"Final validation {epoch+1}/{args.max_epochs}, "
                    f"Val_acc: {val_acc:.4f}, time {val_time:.2f}s"
                    f"{metrics_str}\n"
                    f"*========================================*"
                )
                print(msg)
                if logger:
                    logger.info(msg)
                
                # Check new best
                if val_acc > val_acc_max:
                    print(f"New best ({val_acc_max:.6f} --> {val_acc:.6f})")
                    if logger:
                        logger.info(f"New best ({val_acc_max:.6f} --> {val_acc:.6f})")
                    
                    val_acc_max = val_acc
                    best_metrics = metrics
                    is_new_best = True
                    
                    # Salva plot del best model
                    class_names = [f"class {i}" for i in range(cm.shape[0])]
                    plot_confusion_matrix(
                        cm, 
                        class_names=class_names,
                        title=f'Confusion Matrix - Epoch {epoch+1}',
                        save_path=os.path.join(cm_plots_dir, f"best_confusion_matrix.png")
                    )
                    plot_confusion_matrix(
                        train_cm,
                        class_names=class_names,
                        title=f'Confusion Matrix (Train) - Epoch {epoch+1} ',
                        save_path=os.path.join(cm_plots_dir, f"best_confusion_train_matrix.png")
                    )
                    plot_metrics_table(
                        metrics,
                        class_names=class_names,
                        title=f'Metrics Table - Epoch {epoch+1}',
                        save_path=os.path.join(metrics_plots_dir, f"best_metrics_table.png")
                    )
                    plot_metrics_table(
                        train_metrics,
                        class_names=class_names,
                        title=f'Metrics Table (Train) - Epoch {epoch+1}',
                        save_path=os.path.join(metrics_plots_dir, f"best_train_metrics_table.png")
                    )
                    with open(os.path.join(errors_log_dir, f"best_validation_errors.txt"), 'w') as f:
                        f.write(f"Epoch {epoch+1}:\n")
                        for path, pred, target in val_errors_paths:
                            f.write(f"{path}\tPred: {pred}\tTarget: {target}\n")
                        f.write(f"*" + "-"*40 + "*\n")
                        f.write("\n")

                # Telegram notification
                if use_telegram:
                    telegram_msg = (
                        f"*✅ Validation - Epoch {epoch+1}/{args.max_epochs}*\n"
                        f"Val Acc: {val_acc:.4f}\n"
                        f"Best Val Acc: {val_acc_max:.4f}"
                    )
                    _send_telegram_safe(args, telegram_msg)
            
            # Salva checkpoint
            if should_save:
                save_checkpoint(
                    model, epoch, args, 
                    best_acc=val_acc_max, 
                    optimizer=optimizer, 
                    scheduler=scheduler,
                    filename="model_final.pt"
                )
                
                if is_new_best:
                    print("Copying best model to best_model.pt")
                    if logger:
                        logger.info("Copying best model to best_model.pt")
                    shutil.copyfile(
                        os.path.join(args.final_output_dir, "model_final.pt"),
                        os.path.join(args.final_output_dir, "best_model.pt")
                    )
            
            # Early stopping per validation
            if early_stopping_val and early_stopping_val.step(val_acc, model):
                print(f"[EarlyStopping] Stopping training for val accuracy at epoch {epoch+1}")
                if logger:
                    logger.info(f"[EarlyStopping] Stopping training for val accuracy at epoch {epoch+1}")
                if use_telegram:
                    _send_telegram_safe(args, f"*🛑 Early Stopping (Validation) at Epoch {epoch+1}*")
                break
                        # Early stopping per loss
            if early_stopping_loss and early_stopping_loss.step(val_loss, model):
                print(f"[EarlyStopping] Stopping Validation for loss at epoch {epoch+1}")
                if logger:
                    logger.info(f"[EarlyStopping] Stopping Validation for loss at epoch {epoch+1}")
                if use_telegram:
                    _send_telegram_safe(args, f"*🛑 Early Stopping (Loss) at Epoch {epoch+1}*")
                break
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Fine training
    if is_main_process:
        print(f"Training Finished! Best Accuracy: {val_acc_max:.4f}")
        print("=" * 100)
        print()
        if logger:
            logger.info(f"Training Finished! Best Accuracy: {val_acc_max:.4f}")
            logger.info("=" * 100)
            logger.info("")
        
        if use_telegram:
            time_str = time.strftime('%Y/%m/%d %H:%M')
            telegram_msg = (
                f"*🏆 Training Finished!*\n"
                f"{time_str}\n"
                f"Best Val Acc: {val_acc_max:.4f}"
            )
            _send_telegram_safe(args, telegram_msg)
        
        # Salva plot finali
        if last_cm is not None and last_metrics is not None:
            print(f"Saving plots to: {final_plots_dir}")
            
            class_names = [f"class {i}" for i in range(last_cm.shape[0])]
            
            # Confusion matrix e metrics
            plot_confusion_matrix(
                last_cm,
                class_names=class_names,
                title='Confusion Matrix - Final Epoch',
                save_path=os.path.join(cm_plots_dir, "final_confusion_matrix.png")
            )
            plot_confusion_matrix(
                last_train_cm,
                class_names=class_names,
                title='Confusion Matrix (Train) - Final Epoch ',
                save_path=os.path.join(cm_plots_dir, "final_confusion_train_matrix.png")
            )
            plot_metrics_table(
                last_metrics,
                class_names=class_names,
                title='Metrics Table - Final Epoch',
                save_path=os.path.join(metrics_plots_dir, "final_metrics_table.png")
            )
            plot_metrics_table(
                last_train_metrics,
                class_names=class_names,
                title='Metrics Table (Train) - Final Epoch',
                save_path=os.path.join(metrics_plots_dir, "final_train_metrics_table.png")
            )
            
            # Training curves
            plot_training_curve(
                training_losses,
                metric_name="Loss",
                title="Training Curve - Loss",
                save_path=os.path.join(metrics_plots_dir, "training_loss_curve.png")
            )
            plot_training_curve(
                lr_history,
                metric_name="Learning Rate",
                title="Training Curve - Learning Rate",
                save_path=os.path.join(metrics_plots_dir, "learning_rate_curve.png")
            )
            plot_loss_lr(
                training_losses,
                lr_history,
                title="Training Curve - Loss vs Learning Rate",
                save_path=os.path.join(metrics_plots_dir, "loss_vs_lr_curve.png")
            )
            plot_multi_class_training_curve(
                training_accuracies,
                training_per_class_accuracies,
                title="Training Curve - Accuracy",
                save_path=os.path.join(metrics_plots_dir, "training_accuracy_curve.png")
            )
            plot_multi_class_training_curve(
                validation_accuracies,
                validation_per_class_accuracies,
                title="Validation Curve - Accuracy",
                save_path=os.path.join(metrics_plots_dir, "validation_accuracy_curve.png")
            )
            
            # Telegram plot notifications
            if use_telegram:
                _send_telegram_plots(args, metrics_plots_dir, cm_plots_dir)
    
    return train_loss,train_acc,val_acc_max, best_metrics

def run_testing(
    model,
    test_loader,
    acc_func,
    loss_func,
    args,
    writer_dict=None,
    final_output_dir=None,
    logger=None,
) -> tuple[float, dict]:
    """
    Loop di training principale con validation, early stopping e logging.
    
    Returns:
        float: Best validation accuracy raggiunta
    """
    # Setup logging e writer
    writer = writer_dict.get("writer") if writer_dict is not None else None

    # Inizializza lo step
    args.step= (args.roi_z, int(args.roi_y * 2 // 3), int(args.roi_x * 2 // 3))
    
    # Setup directory output
    final_output_dir = final_output_dir + "/testing"
    args.final_output_dir = final_output_dir 
    
    if final_output_dir is None:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        name_file = f'{args.logdir}_{time_str}'
        final_output_dir = os.path.join(args.output_dir, name_file)
    
    # Crea struttura directory
    final_plots_dir = os.path.join(final_output_dir, "plots")
    cm_plots_dir = os.path.join(final_plots_dir, "confusion_matrix")
    metrics_plots_dir = os.path.join(final_plots_dir, "metrics_tables")
    errors_log_dir = os.path.join(final_output_dir, "errors_logs")
    
    os.makedirs(final_plots_dir, exist_ok=True)
    os.makedirs(cm_plots_dir, exist_ok=True)
    os.makedirs(metrics_plots_dir, exist_ok=True)
    os.makedirs(errors_log_dir, exist_ok=True)

    args.final_plots_dir = final_plots_dir
    
    # Cache attributi comuni
    is_main_process = args.rank == 0
    use_telegram = args.telegram_log if hasattr(args, 'telegram_log') else False
    
    last_cm = None
    last_metrics = None


    if args.distributed:
        torch.distributed.barrier()
    
    epoch_time = time.time()
    if args.patch_merging:
        _,test_acc, test_per_class, cm, test_errors_paths = test_epoch_pm(
            model, test_loader, epoch=0, acc_func=acc_func,loss_func=loss_func, args=args
        )
    else:
        _,test_acc, test_per_class, cm, test_errors_paths = test_epoch(
            model, test_loader, epoch=0, acc_func=acc_func,loss_func=loss_func, args=args
        )

    with open(os.path.join(errors_log_dir, f"testing_errors.txt"), 'a') as f:
        f.write(f"Epoch 0:\n")
        for path, pred, target in test_errors_paths:
            f.write(f"{path}\tPred: {pred}\tTarget: {target}\n")
        f.write(f"*" + "-"*40 + "*\n")
        f.write("\n")

    last_cm = cm
    metrics = metrics_from_confusion_matrix(cm)
    last_metrics = metrics
    metrics_str = format_print_metrics(metrics)
    
    
    if is_main_process:
        val_time = time.time() - epoch_time
        msg = (
            f"Final testing , "
            f"Test_acc: {test_acc:.4f}, time {val_time:.2f}s"
            f"{metrics_str}\n"
            f"*========================================*"
        )
        print(msg)
        if logger:
            logger.info(msg)
        
        # Telegram notification
        if use_telegram:
            telegram_msg = (
                f"*✅ Testing - Epoch 0*\n"
                f"Test Acc: {test_acc:.4f}\n"
            )
            _send_telegram_safe(args, telegram_msg)
            


    
    # Fine training
    if is_main_process:
        print(f"Testing Finished! Best Accuracy: {test_acc:.4f}")
        if logger:
            logger.info(f"Testing Finished! Best Accuracy: {test_acc:.4f}")
            logger.info("=" * 100)
        
        if use_telegram:
            time_str = time.strftime('%Y/%m/%d %H:%M')
            telegram_msg = (
                f"*🏆 Testing Finished!*\n"
                f"{time_str}\n"
                f"Best Test Acc: {test_acc:.4f}"
            )
            _send_telegram_safe(args, telegram_msg)
        
        # Salva plot finali
        if last_cm is not None and last_metrics is not None:
            print(f"Saving plots to: {final_plots_dir}")
            
            class_names = [f"class {i}" for i in range(last_cm.shape[0])]
            
            # Confusion matrix e metrics
            plot_confusion_matrix(
                last_cm,
                class_names=class_names,
                title='Confusion Matrix - Final Epoch',
                save_path=os.path.join(cm_plots_dir, "final_confusion_matrix.png")
            )

            plot_metrics_table(
                last_metrics,
                class_names=class_names,
                title='Metrics Table - Final Epoch',
                save_path=os.path.join(metrics_plots_dir, "final_metrics_table.png")
            )
            
            # Telegram plot notifications
            if use_telegram:
                    _send_telegram_plots_testing(args, metrics_plots_dir, cm_plots_dir)
    
    return test_acc,last_metrics

def run_mc_droput_testing(
    model,
    test_loader,
    acc_func,
    loss_func,
    args,
    writer_dict=None,
    final_output_dir=None,
    logger=None,
) -> tuple[float, dict]:
    """
    Loop di training principale con validation, early stopping e logging.
    
    Returns:
        float: Best validation accuracy raggiunta
    """

    # ============================================
    # Testing con MC-Dropout
    # ============================================
    logger.info("Starting MC-Dropout Testing...")
    test_loss, test_acc, test_acc_per_class, cm, uncertainties = test_epoch_mc_dropout(
        model,
        test_loader,
        epoch=0,
        acc_func=acc_func,
        loss_func=loss_func,
        args=args,
        num_mc_samples=50,  # Numero di MC samples
        logger=logger,
    )

    last_cm = cm
    metrics = metrics_from_confusion_matrix(cm)

    

    return test_loss, test_acc, test_acc_per_class, metrics, uncertainties





import logging
import os
import time
import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast

from config import parse_args
from data_helper import create_dataloaders
from model import UniBertMultiModal
from model import DualBertMultiModal
from model import LXMERTMultiModal
from model import ALBEFMultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from callback.adversarial import FGM, PGD
from callback.ema import EMA
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args, choose_fold):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args, choose_fold)

    # 2. build model and optimizers
    if args.model_type == "unibert":
        model = UniBertMultiModal(args)
    elif args.model_type == "albef":
        model = ALBEFMultiModal(args)
    elif args.model_type == "dualbert":
        model = DualBertMultiModal(args)
    elif args.model_type == "lxmert":
        model = LXMERTMultiModal(args)

    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    if args.fp16:
        scaler = GradScaler()
    
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    early_stop = 0
    
    # fgm and ema
    if args.do_fgm:
        fgm = FGM(model, epsilon=0.5, emb_name='word_embeddings.')
    if args.do_pgd:
        pgd = PGD(model, emb_name='word_embeddings.', epsilon=0.5, alpha=0.3)
    if args.do_ema:
        ema = EMA(model, 0.999)
        ema.register()
    
    for epoch in range(args.max_epochs):
        avg_loss = []
        avg_acc = []
        
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in bar:
            
            model.train()
            if args.fp16:
                with autocast():
                    loss, accuracy, _, _ = model(batch)
            else:
                loss, accuracy, _, _ = model(batch)

            loss = loss.mean()
            avg_loss.append(loss.item())
            
            accuracy = accuracy.mean()
            avg_acc.append(accuracy.item())
            
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if args.do_fgm:
                fgm.attack()
                if args.fp16:
                    with autocast():
                        loss_adv, accuracy_adv, _, _ = model(batch)
                else:
                    loss_adv, accuracy_adv, _, _ = model(batch)

                loss_adv = loss_adv.mean()
                if args.fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    adv_loss.backward() 
                fgm.restore()
                
            if args.do_pgd:
                K = 3
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                        
                    if args.fp16:
                        with autocast():
                            loss_adv, accuracy_adv, _, _ = model(batch)
                    else:
                        loss_adv, accuracy_adv, _, _ = model(batch)

                    loss_adv = loss_adv.mean()
                    if args.fp16:
                        scaler.scale(loss_adv).backward()
                    else:
                        adv_loss.backward() 
                pgd.restore() # 恢复embedding参数
                                        
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad()
            scheduler.step()
            
            if args.do_ema:
                ema.update()
                    
            step += 1
            bar.set_postfix(Epoch=epoch, 
                            Step=step, 
                            loss=np.mean(avg_loss[-100:]), 
                            LR=optimizer.param_groups[0]['lr'], 
                            accuracy=np.mean(avg_acc[-100:]))
        if args.do_ema:
            ema.apply_shadow()
        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1}, f'{args.savedmodel_path}/{args.final_ckpt_file}_fold{choose_fold}.bin')
        if mean_f1 > best_score:
            early_stop = 0
            best_score = mean_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1}, f'{args.savedmodel_path}/{args.ckpt_file}_fold{choose_fold}.bin')
        else:
            early_stop += 1
            if early_stop >= args.patience:
                print("Early Stop")
                break
                
        if args.do_ema:
            ema.restore()
            
def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    start_fold = args.start_fold
    fold = int(1 / args.val_ratio)
    for i in range(fold):
        if i >= start_fold:
            print(f"fold{i} training")
            train_and_validate(args, i)

if __name__ == '__main__':
    main()
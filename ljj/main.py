import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from transformers import BertModel, BertConfig
from model import VisualBertMultiModal, UniBertMultiModal, VisualBertNeXtVLADMultiModal, VlBertMultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from callback.adversarial import FGM, PGD
from callback.ema import EMA
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from tools.common import seed_everything
from tools.common import init_logger, logger


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


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    
    # 2. build model and optimizers
    #model = UniBert_1(args, config)
    model = VisualBertMultiModal(args)

    #model = UniBertMultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    early_stop = 0

    if args.fp16:
        scaler = GradScaler()
    # fgm and ema
    if args.do_fgm:
        fgm = FGM(model, epsilon=args.fgm_epsilon, emb_name='word_embeddings.')
    if args.do_pgd:
        pgd = PGD(model, emb_name='word_embeddings.', epsilon=1, alpha=0.3)
    if args.do_ema:
            ema = EMA(model, 0.999)
            ema.register()
    for epoch in range(args.max_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in bar:
            model.train()
            if args.fp16:
                with autocast():
                    loss, accuracy, _, _ = model(batch)
            else:
                loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
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
                    loss_adv, _, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                if args.fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()
                fgm.restore()
                
            if args.do_pgd:
                #model.zero_grad()
                K = 3
                pgd.backup_grad()
                # 对抗训练
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
                        loss_adv, _, _, _ = model(batch)
                    loss_adv = loss_adv.mean()
                    if args.fp16:
                        scaler.scale(loss_adv).backward()
                    else:
                        loss_adv.backward()
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
                            loss=loss.item(),
                            LR=optimizer.param_groups[0]['lr'], 
                            accuracy=accuracy.item()
                            )
        if args.do_ema:
            ema.apply_shadow()
        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            early_stop = 0
            best_score = mean_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                      os.path.join(args.savedmodel_path, f'visual_roberta_best_mean_f1_{mean_f1}.bin'))
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

    train_and_validate(args)


if __name__ == '__main__':
    main()

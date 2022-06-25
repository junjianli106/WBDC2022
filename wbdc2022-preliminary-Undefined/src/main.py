import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from transformers import BertModel, BertConfig
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from adversarial import FGM, PGD
from ema import EMA
from tqdm import tqdm
from model import WCUniModel

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast


def seed_everything(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask = batch['text_input'], batch['text_mask']
            visual_embeds, visual_attention_mask, val_labels = batch['frame_input'], batch['frame_mask'], batch['label']
            loss, _, pred_label_id, label = model(visual_embeds, visual_attention_mask, input_ids, attention_mask, val_labels)
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
    if args.only_training:
        train_dataloader = create_dataloaders(args, choose_fold)
    else:
        train_dataloader, val_dataloader = create_dataloaders(args, choose_fold)
    # 2. build model and optimizers
   
    model = WCUniModel(args, task=['cls'])

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
    if args.ema:
        ema = EMA(model, 0.999)
        ema.register()
    if args.fgm:
        # 1。5 ：680 6684
        # 1
        fgm = FGM(model, epsilon=1.5, emb_name='bert.embeddings.word_embeddings')
    if args.pgd:
        pgd = PGD(model, emb_name='word_embeddings.', epsilon=1, alpha=0.3)
        
    for epoch in range(args.max_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in bar:
            model.train()
            if args.fp16:
                with autocast():
                    input_ids, attention_mask = batch['text_input'], batch['text_mask']
                    visual_embeds, visual_attention_mask = batch['frame_input'], batch['frame_mask']
                    labels = batch['label']
                    loss, accuracy, _, _ = model(visual_embeds, visual_attention_mask, input_ids, attention_mask, labels)
            else:
                input_ids, attention_mask, token_type_ids = batch['text_input'], batch['text_mask']
                visual_embeds, visual_attention_mask = batch['frame_input'], batch['frame_mask']
                labels = batch['label']
                loss, accuracy, _, _ = model(visual_embeds, visual_attention_mask, input_ids, attention_mask, labels)
            loss = loss.mean()
            accuracy = accuracy.mean()
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if args.fgm:
                fgm.attack()
                if args.fp16:
                    with autocast():
                        input_ids, attention_mask = batch['text_input'], batch['text_mask']
                        visual_embeds, visual_attention_mask = batch['frame_input'], batch['frame_mask']
                        labels = batch['label']
                        loss_adv, accuracy_adv, _, _ = model(visual_embeds, visual_attention_mask, input_ids, attention_mask, labels)
                else:
                    input_ids, attention_mask = batch['text_input'], batch['text_mask']
                    visual_embeds, visual_attention_mask = inputs['frame_input'], inputs['frame_mask']
                    labels = batch['label']
                    loss_adv, accuracy_adv, _, _ = model(visual_embeds, visual_attention_mask, input_ids, attention_mask, labels)
                loss_adv = loss_adv.mean()
                if args.fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()
                fgm.restore()
                    
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()   
            optimizer.zero_grad()
            scheduler.step()
            if args.ema:
                ema.update()
                    
            step += 1
            bar.set_postfix(Epoch=epoch,
                            Step=step,
                            loss=loss.item(),
                            LR=optimizer.param_groups[0]['lr'], 
                            accuracy=accuracy.item()
                            )
        if args.ema:
            ema.apply_shadow()
            
        # 4. validation
        if not args.only_training:
            loss, results = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            if mean_f1 > best_score:
                early_stop = 0
                best_score = mean_f1
                torch.save({'epoch': epoch, 'mean_f1': mean_f1, 'model_state_dict': model.module.state_dict()},
                          os.path.join(args.savedmodel_path, f'Unibert_f1_fold_{choose_fold}.bin'))
            else:
                early_stop += 1
                if early_stop >= args.patience:
                    print("Early Stop")
                    break
        else:
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                          os.path.join(args.savedmodel_path, f'Unibert_fold_{choose_fold}_epoch_{epoch}.bin'))
                
        if args.ema:
            ema.restore()
            
def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    
    if not args.only_training:
        start_fold = args.start_fold
        #start_fold = 2
        fold = int(1 / args.val_ratio)
        for i in range(fold):
            print(f'FOlD :{i}')
            if i >= start_fold:
                print(f"fold{i} training")
                train_and_validate(args, i)
        #train_and_validate(args, 0)
    else:
        train_and_validate(args, -1)
    #train_and_validate(args, 0)

if __name__ == '__main__':
    main()

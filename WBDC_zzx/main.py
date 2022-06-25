import logging
import os
import time
import torch
import datetime

import ray
from ray.tune import CLIReporter
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from visualbert_model import MyVisualBert

from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from adversarial import FGM, PGD
from ema_swa import EMA



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
    if args.only_train:
        train_dataloader = create_dataloaders(args)
    else:
        train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    if args.model_type == 'vbert':
        model = MyVisualBert(args)
        for name, par in model.named_parameters():
            print(name)
        
        if args.pretrain_path:
            print(f'load pretrain ckpt from: {args.pretrain_path}')
            ckpt = torch.load(args.pretrain_path, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f'missing keys:{missing_keys}, unexpected_keys:{unexpected_keys}')
            
    else:
        model = MultiModal(args)
        
    # 计算最大步数
    args.max_steps = args.max_epochs * len(train_dataloader)
    print(f'max_steps:{args.max_steps}')
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    
    if args.ema:
        ema = EMA(model, decay=args.ema)
        # ema.register()
        
    
    if args.fgm:
        fgm = FGM(args, model)
    
    if args.pgd:
        pgd = PGD(args, model)
        

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        if epoch == args.ema_start_epoch and args.ema:
            ema.register()
            
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            if args.fgm:
                fgm.attack()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                fgm.restore()
                loss.backward()
                
            elif args.pgd:
                pgd.backup_grad()
                K = 3
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0))
                    if t != K-1:
                        optimizer.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss, accuracy, _, _ = model(batch)
                    loss = loss.mean()
                    loss.backward()
                
                pgd.restore()
                
            optimizer.step()
            if args.ema and epoch >= args.ema_start_epoch:
                ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                # remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                remaining_time = datetime.timedelta(seconds=remaining_time)
                
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        if not args.only_train:
            if args.ema and epoch >= args.ema_start_epoch:
                ema.apply_shadow()
            loss, results = validate(model, val_dataloader)
            if args.ema and epoch >= args.ema_start_epoch:
                ema.restore()
                
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            if mean_f1 > best_score:
                best_score = mean_f1
                a1 = results['lv1_f1_macro']
                i1 = results['lv1_f1_micro']
                a2 = results['lv2_f1_macro']
                i2 = results['lv2_f1_micro']
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.apply_shadow()
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                        f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}_a1_{a1}_i1_{i1}_a2_{a2}_i2_{i2}.bin')
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.restore()
                    
        else:
            if args.ema and epoch >= args.ema_start_epoch:
                ema.apply_shadow()
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                    f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
            if args.ema and epoch >= args.ema_start_epoch:
                ema.restore()
                


def tune_par(args):
    
    def train_val_tune(config):
        os.chdir('/data/zhangzhexin/weixin_2022/challenge/')
        print(f'tune config:{config}')
        config_str = []
        
        for k, v in config.items():
            args.__setattr__(k, v)
            config_str.append(f'{k}{v}')
        
        setup_seed(args)
            
        save_dir = os.path.join(args.savedmodel_path, '_'.join(config_str))
        print(f'save dir:{save_dir}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 1. load data
        if args.only_train:
            train_dataloader = create_dataloaders(args)
        else:
            train_dataloader, val_dataloader = create_dataloaders(args)

        # 2. build model and optimizers
        if args.model_type == 'vbert':
            model = MyVisualBert(args)
            for name, par in model.named_parameters():
                print(name)
            
            if args.pretrain_path:
                print(f'load pretrain ckpt from: {args.pretrain_path}')
                ckpt = torch.load(args.pretrain_path, map_location='cpu')
                missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
                print(f'missing keys:{missing_keys}, unexpected_keys:{unexpected_keys}')
                
        else:
            model = MultiModal(args)
            
        # 计算最大步数
        args.max_steps = args.max_epochs * len(train_dataloader)
        print(f'max_steps:{args.max_steps}')
        optimizer, scheduler = build_optimizer(args, model)
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))
        
        if args.ema:
            ema = EMA(model, decay=args.ema)
            # ema.register()
            
        
        if args.fgm:
            fgm = FGM(args, model)
        
        if args.pgd:
            pgd = PGD(args, model)
            

        # 3. training
        step = 0
        best_score = args.best_score
        start_time = time.time()
        num_total_steps = len(train_dataloader) * args.max_epochs
        for epoch in range(args.max_epochs):
            if epoch == args.ema_start_epoch and args.ema:
                ema.register()
                
            for batch in train_dataloader:
                model.train()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
                if args.fgm:
                    fgm.attack()
                    loss, accuracy, _, _ = model(batch)
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                    fgm.restore()
                    loss.backward()
                    
                elif args.pgd:
                    pgd.backup_grad()
                    K = 3
                    for t in range(K):
                        pgd.attack(is_first_attack=(t==0))
                        if t != K-1:
                            optimizer.zero_grad()
                        else:
                            pgd.restore_grad()
                        loss, accuracy, _, _ = model(batch)
                        loss = loss.mean()
                        loss.backward()
                    
                    pgd.restore()
                    
                optimizer.step()
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.update()
                optimizer.zero_grad()
                scheduler.step()

                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    # remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    remaining_time = datetime.timedelta(seconds=remaining_time)
                    
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            if not args.only_train:
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.apply_shadow()
                loss, results = validate(model, val_dataloader)
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.restore()
                    
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

                # 5. save checkpoint
                mean_f1 = results['mean_f1']
                print(f'mean f1: {mean_f1} best_score: {best_score}')
                if mean_f1 > best_score:
                    best_score = mean_f1
                    a1 = results['lv1_f1_macro']
                    i1 = results['lv1_f1_micro']
                    a2 = results['lv2_f1_macro']
                    i2 = results['lv2_f1_micro']
                    if args.ema and epoch >= args.ema_start_epoch:
                        ema.apply_shadow()
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                            f'{save_dir}/model_epoch_{epoch}_mean_f1_{mean_f1}_a1_{a1}_i1_{i1}_a2_{a2}_i2_{i2}.bin')
                    if args.ema and epoch >= args.ema_start_epoch:
                        ema.restore()
                
                tune.report(mean_f1=best_score, a1=a1, i1=i1, a2=a2, i2=i2)

                
            else:
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.apply_shadow()
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                        f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.restore()

    
    
    # config = {
    #     'learning_rate': tune.grid_search([2e-5, 5e-5, 8e-5, 1e-4]),
    #     'warmup_ratio': tune.grid_search([0, 0.05, 0.1, 0.15])
    # }    
    config = {
        # 'learning_rate': tune.grid_search([5e-5, 6e-5, 7e-5, 8e-5, 9e-5]),
        # 'warmup_ratio': tune.grid_search([0.1]),
        # 'text_fgm_eps': tune.grid_search([0.15, 0.2, 0.14, 0.16, 0.17, 0.18]),
        'bert_seq_length': tune.grid_search([192]),
        'seed': tune.grid_search([2022]),
        # 'text_pgd_alpha': tune.grid_search([0.05, 0.1, 0.15, 0.2]),
        # 'pgd_epsilon': tune.grid_search([0.2, 0.15, 0.25])
        'text_fgm_eps': tune.grid_search([0.15, 0.2, 0.25]),
        # 'max_epochs': tune.grid_search([4])
        # 'learning_rate': tune.grid_search([5e-5, 6e-5, 7e-5, 8e-5]),
        # 'warmup_ratio': tune.grid_search([0.05, 0.1, 0.15])
        # 'ema': tune.grid_search([0.9999])
        # 'lr_decay': tune.grid_search(['linear', 'cos', 'poly'])
        # 'ema_start_epoch': tune.grid_search([1, 2])
        # 'text_pgd_alpha': tune.grid_search([0.2, 0.5, 0.8]),
        # 'pgd_epsilon': tune.grid_search([0.2, 0.5, 0.8])
    }
    
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["mean_f1", "a1", "i1", "a2", "i2", "training_iteration"],
        max_report_frequency=20)
    
    result = tune.run(
        train_val_tune,
        resources_per_trial={'cpu':4, 'gpu':1},
        config=config,
        metric='mean_f1',
        mode='max',
        progress_reporter=reporter,
        num_samples=1,
    )
    
    best_trial = result.get_best_trial('mean_f1')
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final mean f1: {}".format(
        best_trial.last_result["mean_f1"]))

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    if args.tune:
        tune_par(args)
    else:
        train_and_validate(args)


if __name__ == '__main__':
    main()

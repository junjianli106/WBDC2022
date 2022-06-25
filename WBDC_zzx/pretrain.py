import logging
import os
import time
import torch
import datetime
from accelerate import Accelerator

import ray
from ray.tune import CLIReporter
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from pretrain_model import MyVisualBert

from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from adversarial import FGM
from ema_swa import EMA
from tqdm import tqdm


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    each_loss_dict = {'mlm': [], 'itm': [], 'mfm': []}
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            loss, mlm_loss, itm_loss, mfm_loss = model(batch)
            # if each_loss_dict is None:
            #     each_loss_dict = {k: [] for k in each_loss.keys()}
            loss = loss.mean()
            losses.append(loss.cpu().numpy())
            mlm_loss = mlm_loss.mean()
            itm_loss = itm_loss.mean()
            mfm_loss = mfm_loss.mean()
            each_loss_dict['mlm'].append(mlm_loss.cpu().numpy())
            each_loss_dict['itm'].append(itm_loss.cpu().numpy())
            each_loss_dict['mfm'].append(mfm_loss.cpu().numpy())
            # break
            
    loss = sum(losses) / len(losses)
    for k in each_loss_dict.keys():
        v = each_loss_dict[k]
        each_loss_dict[k] = sum(v) / len(v)

    model.train()
    return loss, each_loss_dict


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
        
    else:
        model = MultiModal(args)
        
    # 计算最大步数
    args.max_steps = args.max_epochs * len(train_dataloader)
    print(f'max_steps:{args.max_steps}')
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        # accelerator = Accelerator()
    
    if args.ema:
        ema = EMA(model)
        ema.register()
        
    
    if args.fgm:
        fgm = FGM(args, model)
    
        

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    
    # model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, mlm_loss, itm_loss, mfm_loss = model(batch)
            loss = loss.mean()
            mlm_loss = mlm_loss.mean()
            itm_loss = itm_loss.mean()
            mfm_loss = mfm_loss.mean()
            loss.backward()
            # accelerator.backward(loss)
            if args.fgm:
                fgm.attack()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                fgm.restore()
                loss.backward()
                
            optimizer.step()
            if args.ema:
                ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                # remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                remaining_time = datetime.timedelta(seconds=remaining_time)

                logging.info(f"Epoch {epoch} step/total {step}/{num_total_steps} eta {remaining_time}: loss {loss:.3f}, mlm loss {mlm_loss:.3f} ,itm loss {itm_loss:.3f}, mfm loss {mfm_loss:.3f}")

        # 4. validation
        if not args.only_train:
            if args.ema:
                ema.apply_shadow()
            loss, each_loss_dict = validate(model, val_dataloader)
            if args.ema:
                ema.restore()
                
            # 5. save checkpoint
            logging.info(f'epoch {epoch}, validation loss: {loss} {each_loss_dict}')
            if args.ema:
                ema.apply_shadow()
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            name = f'model_epoch_{epoch}_loss_{loss}'
            for k, v in each_loss_dict.items():
                name += f'_{k}_{v:.3f}'
            torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                    f'{args.savedmodel_path}/{name}')
            if args.ema:
                ema.restore()
                    
        else:
            if args.ema:
                ema.apply_shadow()
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                    f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
            if args.ema:
                ema.restore()
                


def tune_par(args):
    
    def train_val_tune(config):
        os.chdir('/data/zhangzhexin/weixin_2022/challenge/')
        print(f'tune config:{config}')
        config_str = []
        
        for k, v in config.items():
            args.__setattr__(k, v)
            config_str.append(f'{k}{v}')
            
        save_dir = os.path.join(args.savedmodel_path, '_'.join(config_str))
        print(f'save dir:{save_dir}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 1. load data
        train_dataloader, val_dataloader = create_dataloaders(args)

        # 2. build model and optimizers
        if args.model_type == 'vbert':
            model = MyVisualBert(args)
            for name, par in model.named_parameters():
                print(name)
            
        else:
            model = MultiModal(args)
            
        # 计算最大步数
        args.max_steps = args.max_epochs * len(train_dataloader)
        print(f'max_steps:{args.max_steps}')
        optimizer, scheduler = build_optimizer(args, model)
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))
        
        if args.ema:
            ema = EMA(model)
            ema.register()
            
        
        if args.fgm:
            fgm = FGM(args, model)
        
            

        # 3. training
        step = 0
        best_score = args.best_score
        start_time = time.time()
        num_total_steps = len(train_dataloader) * args.max_epochs
        for epoch in range(args.max_epochs):
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
                    
                optimizer.step()
                if args.ema:
                    ema.update()
                optimizer.zero_grad()
                scheduler.step()

                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            if args.ema:
                ema.apply_shadow()
            loss, results = validate(model, val_dataloader)
            if args.ema:
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
                if args.ema:
                    ema.apply_shadow()
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                        f'{save_dir}/model_epoch_{epoch}_mean_f1_{mean_f1}_a1_{a1}_i1_{i1}_a2_{a2}_i2_{i2}.bin')
                if args.ema:
                    ema.restore()

            tune.report(mean_f1=best_score, a1=a1, i1=i1, a2=a2, i2=i2)
    
    
    # config = {
    #     'learning_rate': tune.grid_search([2e-5, 5e-5, 8e-5, 1e-4]),
    #     'warmup_ratio': tune.grid_search([0, 0.05, 0.1, 0.15])
    # }    
    config = {
        'learning_rate': tune.grid_search([5e-5, 6e-5, 7e-5, 8e-5, 9e-5]),
        'warmup_ratio': tune.grid_search([0.1]),
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

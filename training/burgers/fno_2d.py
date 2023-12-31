import torch
import logging
import os
from tqdm import tqdm
from utils import AverageRecord, Metrics, LpLoss
from time import time
from models import FNO2d
from dataset import BurgersDataset
from visualize import burgers_heatmap, burgers_movie
from ..base import BaseTrainer


class FNO2DTrainer(BaseTrainer):
    def __init__(self, model, device, epochs, eval_freq=5, patience=-1,
                 verbose=False, wandb_log=False, logger=False, 
                 saving_best=True, saving_checkpoints=False, saving_path=None):
        super().__init__(
            model=model, device=device, epochs=epochs, eval_freq=eval_freq, patience=patience,
            verbose=verbose, wandb_log=wandb_log, logger=logger,
            saving_best=saving_best, saving_checkpoints=saving_checkpoints, saving_path=saving_path)
    
    def process(self, model, train_loader, valid_loader, test_loader, 
                optimizer, criterion, eval_metrics, regularizer=None, scheduler=None,):
        if self.verbose:
            self.logger("Start training")
            self.logger("Train dataset size: {}".format(len(train_loader.dataset)))
            self.logger("Valid dataset size: {}".format(len(valid_loader.dataset)))
            self.logger("Test dataset size: {}".format(len(test_loader.dataset)))

        best_epoch = 0
        best_metrics = None
        counter = 0
        
        for epoch in range(self.epochs):
            train_loss = self.train(model, train_loader, optimizer, criterion, scheduler, regularizer)
            if self.verbose:
                self.logger("Epoch {} | average train loss: {:.4f} | lr: {:.6f}".format(epoch, train_loss, optimizer.param_groups[0]["lr"]))
            del train_loss
            
            if (epoch + 1) % self.eval_freq == 0:
                valid_loss, valid_metrics = self.evaluate(model, valid_loader, criterion, eval_metrics, split="valid")
                if self.verbose:
                    self.logger("Epoch {} | valid loss: {:.4f} | valid metrics: {}".format(epoch, valid_loss, valid_metrics))
                valid_metrics = valid_metrics.to_dict()
                
                if not best_metrics or valid_metrics[eval_metrics[0]] < best_metrics[eval_metrics[0]]:
                    counter = 0
                    best_epoch = epoch
                    best_metrics = valid_metrics
                    torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
                    model.cuda()
                    if self.verbose:
                        self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                elif self.patience != -1:
                    counter += 1
                    if counter >= self.patience:
                        if self.verbose:
                            self.logger("Early stop at epoch {}".format(epoch))
                        break

        self.logger("Optimization Finished!")

        # load best model
        if not best_metrics:
            torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(self.saving_path, "best_model.pth")))
            self.logger("Load best models at epoch {} from {}".format(best_epoch, self.saving_path))        
        model.cuda()
        
        _, valid_metrics = self.evaluate(model, valid_loader, criterion, eval_metrics, split="valid")
        self.logger("Valid metrics: {}".format(valid_metrics))
        _, test_metrics = self.evaluate(model, test_loader, criterion, eval_metrics, split="test")
        self.logger("Test metrics: {}".format(test_metrics))
        
    def train(self, model, train_loader, optimizer, criterion, scheduler=None, regularizer=None, **kwargs):
        train_loss = AverageRecord()
        model.cuda()
        model.train()
        with tqdm(total=len(train_loader)) as bar:
            for x, y in train_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                # compute loss
                y_pred = model(x).reshape(y.shape)
                data_loss = criterion(y_pred, y)
                loss = data_loss
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record loss and update progress bar
                train_loss.update(loss.item())
                bar.update(1)
                bar.set_postfix_str("train loss: {:.4f}".format(loss.item()))
            if scheduler is not None:
                scheduler.step()
        return train_loss.avg
    
    def evaluate(self, model, eval_loader, criterion, metric_list, split="valid"):
        eval_metric = Metrics(metric_list, split)
        eval_loss = AverageRecord()
        model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                # compute loss
                y_pred = model(x).reshape(y.shape)
                data_loss = criterion(y_pred, y)
                loss = data_loss
                eval_loss.update(loss.item())
                # compute metrics
                eval_metric.update(y_pred.cpu(), y.cpu())
        return eval_loss.avg, eval_metric
    
    def visualize(self, model, dataset, heatmap=False, movie=False):
        for x, y in dataset.test_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            y_pred = model(x).reshape(y.shape)
            break
        
        x = x.cpu().detach().numpy()[0]
        y = y.cpu().detach().numpy()[0]
        y_pred = y_pred.cpu().detach().numpy()[0]
        
        if heatmap:
            burgers_heatmap(y, y_pred, 
                            start_x=dataset.start_x, end_x=dataset.end_x, 
                            dx=dataset.dx, t=dataset.t, dt=dataset.dt, v=dataset.v,
                            file_path=os.path.join(self.saving_path, "burgers_heatmap.png"))
        
        if movie:
            burgers_movie(y, y_pred, 
                          start_x=dataset.start_x, end_x=dataset.end_x, 
                          dx=dataset.dx, t=dataset.t, dt=dataset.dt, v=dataset.v,
                          file_path=os.path.join(self.saving_path, "burgers_movie.gif"))


def fno_burgers(args):
    if args['verbose']:
        logger = logging.info if args['log'] else print
    # load data
    if args['verbose']:
        logger("Loading {} dataset".format(args['dataset']))
    start = time()
    dataset = BurgersDataset(
        data_path=args['data_path'],
        num_grid_x=args['num_grid_x'],
        num_grid_t=args['num_grid_t'],
        x_sample_factor=args['x_sample_factor'],
        t_sample_factor=args['t_sample_factor'],
        v = args['v'],
        start_x=args['start_x'],
        end_x=args['end_x'],
        t=args['t'],
        batch_size=args['batch_size'],
        train_ratio=args['train_ratio'],
        valid_ratio=args['valid_ratio'],
        test_ratio=args['test_ratio'],
        subset=args['subset'],
        subset_ratio=args['subset_ratio'],
    )
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    test_loader = dataset.test_loader
    if args['verbose']:
        logger("Loading data costs {: .2f}s".format(time() - start))
    
    # build model
    if args['verbose']:
        logger("Building models")
    start = time()
    model = FNO2d(
        modes1=args['modes1'],
        modes2=args['modes2'],
        width=args['width'],
        fc_dim=args['fc_dim'],
        layers=args['layers'],
        act=args['act'],
    )
    model = model.to('cuda')                                                   
    optimizer = torch.optim.Adam(
        model.parameters(), 
        betas=(0.9, 0.999),
        lr=args['lr'],
        # weight_decay=args['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args['milestones'],
        gamma=args['gamma'],
    )
    criterion = LpLoss(d=2, p=2)
    eval_metrics = ['l2']
    
    if args['verbose']:
        logger("Model: {}".format(model))
        logger("Optimizer: {}".format(optimizer))
        logger("Scheduler: {}".format(scheduler))
        logger("Criterion: {}".format(criterion))
        logger("Eval metrics: {}".format(eval_metrics))
        logger("Building models costs {: .2f}s".format(time() - start))
    
    trainer = FNO2DTrainer(
        model=model, 
        epochs=args['epochs'],
        device='cuda',
        eval_freq=args['eval_freq'],
        patience=args['patience'],
        verbose=args['verbose'],
        logger=args['log'],
        wandb_log=args['wandb'],
        saving_best=args['saving_best'],
        saving_checkpoints=args['saving_checkpoints'],
        saving_path=args['saving_path'],
    )
    
    trainer.process(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        eval_metrics=eval_metrics,
    )
    
    if args['visualize']:
        trainer.visualize(model, dataset, heatmap=args['heatmap'], movie=args['movie'])

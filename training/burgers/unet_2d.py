import torch
import logging
import os
from tqdm import tqdm
from utils import LpLoss, LossRecord, burgers_loss
from time import time
from models import UNet2d
from dataset import BurgersDataset
from visualize import burgers_heatmap, burgers_movie
from ..base import BaseTrainer


class UNet2DTrainer(BaseTrainer):
    def __init__(self, model, device, epochs, eval_freq=5, patience=-1,
                 verbose=False, wandb_log=False, logger=False, 
                 saving_best=True, saving_checkpoints=False, saving_path=None,
                 data_weight=5.0, f_weight=1.0, ic_weight=1.0):
        super().__init__(
            model=model, device=device, epochs=epochs, eval_freq=eval_freq, patience=patience,
            verbose=verbose, wandb_log=wandb_log, logger=logger,
            saving_best=saving_best, saving_checkpoints=saving_checkpoints, saving_path=saving_path)
        self.data_weight = data_weight
        self.f_weight = f_weight
        self.ic_weight = ic_weight
    
    def process(self, model, train_loader, valid_loader, test_loader, 
                optimizer, criterion, eval_metrics, regularizer=None, scheduler=None, **kwargs):
        if self.verbose:
            self.logger("Start training")
            self.logger("Train dataset size: {}".format(len(train_loader.dataset)))
            self.logger("Valid dataset size: {}".format(len(valid_loader.dataset)))
            self.logger("Test dataset size: {}".format(len(test_loader.dataset)))

        best_epoch = 0
        best_metrics = None
        counter = 0
        
        for epoch in range(self.epochs):
            train_loss_record = self.train(model, train_loader, optimizer, criterion, scheduler, regularizer, v=kwargs['v'], t=kwargs['t'])
            if self.verbose:
                self.logger("Epoch {} | {} | lr: {:.6f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))

            if (epoch + 1) % self.eval_freq == 0:
                valid_loss_record = self.evaluate(model, valid_loader, criterion, eval_metrics, split="valid", v=kwargs['v'], t=kwargs['t'])
                if self.verbose:
                    self.logger("Epoch {} | {}".format(epoch, valid_loss_record))
                valid_metrics = valid_loss_record.to_dict()
                
                if not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']:
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
        
        valid_loss_record = self.evaluate(model, valid_loader, criterion, eval_metrics, split="valid", v=kwargs['v'], t=kwargs['t'])
        self.logger("Valid loss: {}".format(valid_loss_record))
        test_loss_record = self.evaluate(model, test_loader, criterion, eval_metrics, split="test", v=kwargs['v'], t=kwargs['t'])
        self.logger("Test loss: {}".format(test_loss_record))

    def train(self, model, train_loader, optimizer, criterion, scheduler=None, regularizer=None, **kwargs):
        loss_record = LossRecord(['train_loss', 'data_loss', 'equation_loss', 'ic_loss'])
        model.cuda()
        model.train()
        with tqdm(total=len(train_loader)) as bar:
            for x, y in train_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                # pad x
                x_0 = x[:, :1, :, :]
                x_0 = x_0.repeat(1, x.shape[2]-x.shape[1], 1, 1)
                new_x = torch.cat([x_0, x], dim=1)
                # compute loss
                y_pred = model(new_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                y_pred = y_pred[:, -y.shape[1]:, :, 0]
                data_loss = criterion(y_pred, y)
                # cut off the padded part
                ic_loss, equation_loss = burgers_loss(y_pred, x[:, 0, :, 0], v=kwargs['v'], t=kwargs['t'])
                train_loss = self.data_weight * data_loss + self.f_weight * equation_loss + self.ic_weight * ic_loss
                # compute gradient
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                # record loss and update progress bar
                loss_record.update({
                    'train_loss': train_loss.item(),
                    'data_loss': data_loss.item(),
                    'equation_loss': equation_loss.item(),
                    'ic_loss': ic_loss.item(),
                })
                bar.update(1)
                bar.set_postfix_str("train loss: {:.4f}".format(train_loss.item()))
            if scheduler is not None:
                scheduler.step()
        return loss_record
    
    def evaluate(self, model, eval_loader, criterion, metric_list, split="valid", **kwargs):
        loss_record = LossRecord([split + '_loss', 'data_loss', 'equation_loss', 'ic_loss'])
        model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                # pad x
                x_0 = x[:, :1, :, :]
                x_0 = x_0.repeat(1, x.shape[2]-x.shape[1], 1, 1)
                new_x = torch.cat([x_0, x], dim=1)
                # compute loss
                y_pred = model(new_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                # cut off the padded part
                y_pred = y_pred[:, -y.shape[1]:, :, 0]
                data_loss = criterion(y_pred, y)
                ic_loss, equation_loss = burgers_loss(y_pred, x[:, 0, :, 0], v=kwargs['v'], t=kwargs['t'])
                eval_loss = self.data_weight * data_loss + self.f_weight * equation_loss + self.ic_weight * ic_loss
                loss_record.update({
                    split + '_loss': eval_loss.item(),
                    'data_loss': data_loss.item(),
                    'equation_loss': equation_loss.item(),
                    'ic_loss': ic_loss.item(),
                })
        return loss_record
    
    def visualize(self, model, dataset, heatmap=False, movie=False):
        with torch.no_grad():
            for x, y in dataset.test_loader:
                x = x.to('cuda')
                x_0 = x[:, :1, :, :]
                x_0 = x_0.repeat(1, x.shape[2]-x.shape[1], 1, 1)
                new_x = torch.cat([x_0, x], dim=1)
                y = y.to('cuda')
                y_pred = model(new_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                y_pred = y_pred[:, -y.shape[1]:, :, 0]
                break
        
        x = x.cpu().detach().numpy()[0]
        y = y.cpu().detach().numpy()[0]
        y_pred = y_pred.cpu().detach().numpy()[0]
        
        if heatmap:
            burgers_heatmap(y, y_pred, 
                            start_x=dataset.test_dataset.start_x, end_x=dataset.test_dataset.end_x,
                            dx=dataset.test_dataset.dx, t=dataset.test_dataset.t, dt=dataset.test_dataset.dt, v=dataset.test_dataset.v,
                            file_path=os.path.join(self.saving_path, "burgers_heatmap.png"))
        
        if movie:
            burgers_movie(y, y_pred, 
                          start_x=dataset.test_dataset.start_x, end_x=dataset.test_dataset.end_x,
                          dx=dataset.test_dataset.dx, t=dataset.test_dataset.t, dt=dataset.test_dataset.dt, v=dataset.test_dataset.v,
                          file_path=os.path.join(self.saving_path, "burgers_movie.gif"))


def unet_2d_burgers(args):
    if args['verbose']:
        logger = logging.info if args['log'] else print
    # load data
    if args['verbose']:
        logger("Loading {} dataset".format(args['dataset']))
    start = time()
    dataset = BurgersDataset(
        data_path=args['data_path'],
        raw_resolution=args['raw_resolution'],
        sample_resolution=args['sample_resolution'],
        eval_resolution=args['eval_resolution'],
        v = args['v'],
        start_x=args['start_x'],
        end_x=args['end_x'],
        t=args['t'],
        train_batchsize=args['train_batchsize'],
        eval_batchsize=args['eval_batchsize'],
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
    model = UNet2d(
        in_channels=args['in_channels'],
        out_channels=args['out_channels'],
        init_features=args['init_features'],
    )
    model = model.to('cuda')                                                   
    optimizer = torch.optim.Adam(
        model.parameters(), 
        betas=(0.9, 0.999),
        lr=args['lr'],
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
    
    trainer = UNet2DTrainer(
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
        data_weight=args['data_weight'],
        f_weight=args['f_weight'],
        ic_weight=args['ic_weight'],
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
        v=args['v'],
        t=args['t'],
    )
    
    if args['visualize']:
        trainer.visualize(model, dataset, heatmap=args['heatmap'], movie=args['movie'])

import logging
from utils import AverageRecord, Metrics
from tqdm import tqdm
import torch
import os


class BaseTrainer:
    def __init__(self, model, device, epochs, eval_freq=5, patience=-1,
                 verbose=False, wandb_log=False, logger=False, 
                 saving_best=True, saving_checkpoints=False, saving_path=None):
        self.device = device
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.patience = patience
        self.wandb = wandb_log
        self.verbose = verbose
        self.saving_best = saving_best
        self.saving_checkpoints = saving_checkpoints
        self.saving_path = saving_path
        if verbose:
            self.logger = logging.info if logger else print
    
    def train(self, model, train_loader, optimizer, criterion, scheduler=None, regularizer=None):
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

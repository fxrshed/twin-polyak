from abc import ABC, abstractmethod

import torch
from pt_methods import TwinPolyakMA, Momo
import sps
from alig.th import AliG

import lightning as L


optimizers_dict = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'Momo': Momo,
    'SPS': sps.Sps,
    'Adagrad': torch.optim.Adagrad,
    'AliG': AliG
}


class BaseTrainingModule(L.LightningModule, ABC):

    def __init__(self, config: dict):
        super().__init__()

        self.optimizer = config['optimizer']
        self.optimizer_hparams = config['optimizer_hparams']
        self.l2_reg_term = config['reg']

        if self.optimizer == "STP":
            self.automatic_optimization = False
            self.model_x = self.build_model()
            self.model_y = self.build_model()
        else:
            self.model = self.build_model()

        self.loss_fn = self.define_loss_fn()
        self.val_acc = self.define_val_acc_metric()

    @abstractmethod
    def define_loss_fn(self, *args, **kwargs):
        pass

    @abstractmethod
    def define_val_acc_metric(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def unpack_batch(self, batch):
        pass

    def training_step(self, batch):
        x, y = self.unpack_batch(batch)

        if self.optimizer == 'STP':

            optimizer_x = self.optimizers()[0]
            optimizer_y = self.optimizers()[1]

            logits_x = self.model_x(x)
            logits_y = self.model_y(x)

            loss_x = self.loss_fn(logits_x, y)
            loss_y = self.loss_fn(logits_y, y)

            if self.l2_reg_term > 0.0:
                l2_norm_x = sum(p.pow(2.0).sum() for p in self.model_x.parameters())
                loss_x += self.l2_reg_term * l2_norm_x
                
                l2_norm_y = sum(p.pow(2.0).sum() for p in self.model_y.parameters())
                loss_y += self.l2_reg_term * l2_norm_y
            
            optimizer_x.zero_grad()
            optimizer_y.zero_grad()

            h_x = optimizer_x.momentum_step(loss_x)
            h_y = optimizer_y.momentum_step(loss_y)

            if h_x > h_y:
                loss_diff = h_x - h_y
                optimizer_x.step(loss_diff=loss_diff)
                self.log('train_loss', loss_x, on_step=False, on_epoch=True, prog_bar=True)
                self.log('lr', optimizer_x.step_size.item(), prog_bar=True)
            else:
                loss_diff = h_y - h_x
                optimizer_y.step(loss_diff=loss_diff)
                self.log('train_loss', loss_y, on_step=False, on_epoch=True, prog_bar=True)
                self.log('lr', optimizer_y.step_size.item(), on_step=True, on_epoch=False, prog_bar=True)
        else:
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            if self.l2_reg_term > 0.0:
                if self.optimizer == 'SPS':
                    l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                    loss += self.l2_reg_term * l2_norm

            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

            return loss

    def on_train_batch_end(self, *args, **kwargs):
        if isinstance(self.optimizers(), Momo):
            self.log('lr', self.optimizers().state['step_size_list'][-1], on_step=True, on_epoch=False, prog_bar=True)
        elif isinstance(self.optimizers(), sps.Sps):
            self.log('lr', self.optimizers().state['step_size'], on_step=True, on_epoch=False, prog_bar=True)
        elif self.optimizer == "STP":
            pass
        else:
            self.log('lr', self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)

    def validation_step(self, batch):
        x, y = self.unpack_batch(batch)

        if self.optimizer == "STP":
            logits_x = self.model_x(x)
            logits_y = self.model_y(x)

            loss_x = self.loss_fn(logits_x, y)
            loss_y = self.loss_fn(logits_y, y)

            if loss_x < loss_y:
                self.log('val_loss', loss_x, on_step=False, on_epoch=True, prog_bar=True)
                if self.val_acc is not None:
                    self.val_acc(logits_x, y)
                    self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log('val_loss', loss_y, on_step=False, on_epoch=True, prog_bar=True)
                if self.val_acc is not None:
                    self.val_acc(logits_y, y)
                    self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        else:
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

            if self.val_acc is not None:
                self.val_acc(logits, y)
                self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
     
    def configure_optimizers(self):
        optimizer = self.hparams['config']['optimizer']
        optimizer_hparams = self.hparams['config']['optimizer_hparams']
        if optimizer == "STP":
            return [TwinPolyakMA(self.model_x.parameters(), **optimizer_hparams),
                    TwinPolyakMA(self.model_y.parameters(), **optimizer_hparams)]
        else:
            optimizer = optimizers_dict[optimizer](self.model.parameters(), **optimizer_hparams)
            return [optimizer]
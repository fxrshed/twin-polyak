import numpy as np

from npkit.metrics import LogisticRegressionAccuracy
from npkit.loss import LogisticRegressionLoss
from npkit.optim import *
from npkit.trainer import NumpyBaseTrainingModule

class NumpyLinearModel:
    def __init__(self, input_dim: int):
        self.params = np.random.randn(input_dim)

    def __call__(self, X):
        return X @ self.params
    
    def parameters(self):
        return self.params


class NumpyLibSVMBinaryClassifier(NumpyBaseTrainingModule):

    def __init__(self, 
                 input_dim,
                 config,
                 max_epochs):
        super().__init__(max_epochs)

        self.config = config
        self.model = self.build_model(input_dim)
        self.optimizer = self.configure_optimizers()
        self.loss_function = self.define_loss_fn()
        self.metric_accuracy = LogisticRegressionAccuracy()

        if self.config['optimizer'] is STP:
            self.logger.log('model_x', self.model[0].params.copy(), on_epoch=False)
            self.logger.log('model_y', self.model[1].params.copy(), on_epoch=False)

    def define_loss_fn(self):
        return LogisticRegressionLoss(self.config.get('lmd', 0.0))

    def build_model(self, input_dim):
        if self.config['optimizer'] is STP:
            return [
                NumpyLinearModel(input_dim=input_dim),
                NumpyLinearModel(input_dim=input_dim)
            ]
        else:
            return NumpyLinearModel(input_dim=input_dim)

    def training_step(self, batch):
        X, y = batch

        if self.config['optimizer'] is STP:
        
            logits_x = self.model[0](X)
            logits_y = self.model[1](X)

            loss_x = self.loss_function.loss(logits_x, y, w=self.model[0].params)
            grad_x = self.loss_function.grad(logits_x, X, y, w=self.model[0].params)
        
            loss_y = self.loss_function.loss(logits_y, y, w=self.model[1].params)
            grad_y = self.loss_function.grad(logits_y, X, y, w=self.model[1].params)

            h_x = self.optimizer[0].momentum_step(loss_x, grad_x)
            h_y = self.optimizer[1].momentum_step(loss_y, grad_y)
            
            if h_x > h_y:
                loss_diff = h_x - h_y
                self.optimizer[0].step(loss_diff)
                self.logger.log('train_loss', loss_y, on_epoch=True)
                self.logger.log('lr', self.optimizer[0].lr, on_epoch=False)
            else:
                loss_diff = h_y - h_x 
                self.optimizer[1].step(loss_diff)
                self.logger.log('train_loss', loss_x, on_epoch=True)
                self.logger.log('lr', self.optimizer[1].lr, on_epoch=False)

        else:
            logits = self.model(X)

            loss = self.loss_function.loss(logits, y, w=self.model.params)
            grad = self.loss_function.grad(logits, X, y, w=self.model.params)
            
            if self.config['optimizer'] is SLS:
                    def closure(params):
                        return self.loss_function.loss(logits, y, w=params)
                    self.optimizer.step(loss, grad, closure=closure)
            else:
                self.optimizer.step(loss, grad)

            self.logger.log('train_loss', loss, on_epoch=True)
            self.logger.log('lr', self.optimizer.lr, on_epoch=False)
        
    def validation_step(self, batch):
        X, y = batch
        
        if self.config['optimizer'] is STP:
            logits_x = self.model[0](X)
            logits_y = self.model[1](X)

            loss_x = self.loss_function.loss(logits_x, y, w=self.model[0].params)
        
            loss_y = self.loss_function.loss(logits_y, y, w=self.model[1].params)
            
            if loss_x < loss_y:
                loss = loss_x 
                acc = self.metric_accuracy(logits_x, y)
            else:
                loss = loss_y 
                acc = self.metric_accuracy(logits_y, y)

        else:
            logits = self.model(X)

            loss = self.loss_function.loss(logits, y, w=self.model.params)
            acc = self.metric_accuracy(logits, y)
        
        self.logger.log('val_loss', loss)
        self.logger.log('val_acc', acc)

    def configure_optimizers(self):
        optimizer_hparams = self.config['optimizer_hparams']
        print(optimizer_hparams)
        if self.config['optimizer'] is STP:
            return [
                STP(self.model[0].params, **optimizer_hparams),
                STP(self.model[1].params, **optimizer_hparams)
            ]
        else:
            return self.config['optimizer'](self.model.params, **optimizer_hparams)

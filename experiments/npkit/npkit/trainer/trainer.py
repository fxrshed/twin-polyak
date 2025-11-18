from tqdm import tqdm

from .logger import NumpyLogger
from .callback import BaseCallback

class NumpyBaseTrainingModule:

    def __init__(self, 
                 logger: NumpyLogger,
                 max_epochs: int):
        
        self.max_epochs = max_epochs
        
        self.logger = logger

        self.current_epoch = 0
        self.current_step = 0

        self.callbacks: list[BaseCallback] = []

    def build_model(self):
        raise NotImplementedError()

    def define_loss_fn(self):
        raise NotImplementedError()

    def fit(self, datamodule):

        for callback in self.callbacks:
            callback.on_fit_start()
        
        for epoch in tqdm(range(self.max_epochs)):
            
            self.current_epoch = epoch
               
            for batch in datamodule.train_dataloader:
                self.training_step(batch)
                self.current_step += 1

            for batch in datamodule.val_dataloader:
                self.validation_step(batch)

            self.logger.compute_on_epoch(epoch=self.current_epoch, step=self.current_step)
            self.logger.reset()

        for callback in self.callbacks:
            callback.on_fit_end()

    def training_step(self, batch):
        raise NotImplementedError()

    def validation_step(self, batch):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()
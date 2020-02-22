from typing import Callable
from .engine import Engine

class Trainer(Engine):
    """ Warper class for training logic """

    def __init__(self, model, optimizer, loss_function, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.device = device

        self.callbacks_list = {
            'on_batch_start': [],
            'on_batch_end': [],
            'on_epoch_start': [],
            'on_epoch_end': []
        }

        self.is_running = False # Used for early stopping
        self.cache = {} # Used for everyone

    def train_batch(self, train_iter, info={}):
        # Training one batch logic
        X, y = train_iter.next()
        
        X = X.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(X)

        self.optimizer.zero_grad()
        loss = self.loss_function(y_hat, y)
        loss.backward()
        self.optimizer.step()

        # For logging
        info['loss'] = loss.item()
        return info

    def fit(self, train_dataloader, num_epochs=1):
        num_batch = len(train_dataloader)
        self.is_running = True

        for epoch in range(num_epochs):
            self.fire_event('on_epoch_start', epoch=epoch)
            self.model.train()
            
            train_iter = iter(train_dataloader)
            for batch in range(num_batch):
                info = {'batch': batch}
                self.fire_event('on_batch_start', **info)
                info = self.train_batch(train_iter, info)
                self.fire_event('on_batch_end', **info)

            self.fire_event('on_epoch_end', epoch=epoch)

            if not self.is_running:
                return

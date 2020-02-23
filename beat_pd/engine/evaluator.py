from .engine import Engine

from typing import Union, Callable
import numpy as np
import torch
import tqdm

class Evaluator(Engine):
    def __init__(self, model, metrics: dict={}, device='cpu'):
        """ Evaluator Engine class

            model: Model to evaluate
            metrics: Dictionary mapping from metric name 
                     to metric function 
            device: Device for model

        """
        super(Evaluator, self).__init__()
        self.model = model
        self.metrics = metrics
        self.device = device

        self.callbacks_list = {
            'on_eval_start': [],
            'on_eval_end': []
        }

    
    @Engine.add_event_listener("eval")
    def evaluate(self, val_dataloader, *args, **kwargs):
        self.model.to(self.device)
        self.model.eval()

        labels = None
        predict = None

        # Make prediction
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_dataloader, desc='Evaluate'):
                # Convert x and y to gpu if needed
                x = x.to(self.device)
                # y = y.to(self.device)

                y_hat = self.model(x)
                
                if y_hat.is_cuda:
                    # y = y.cpu()
                    y_hat = y_hat.cpu()

                # Stacking and store on RAM
                if labels is not None:
                    labels = torch.cat((labels, y))
                    predict = torch.cat((predict, y_hat))
                else:
                    labels = y
                    predict = y_hat
        
        # Evaluating results
        result = {}
        for metric_name, metric in self.metrics.items():
            result[metric_name] = metric(predict, labels)
        
        result.update(kwargs) # To store num epoch or batch
        return result



            

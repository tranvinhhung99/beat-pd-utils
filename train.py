import random
random.seed(1) # Delete random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from beat_pd.utils import parse_args
from beat_pd.dataset import BeatPD_Dataset
from beat_pd.models.factory import get_model_instance, get_optimizer
from beat_pd.engine import Trainer

def main():
    ## Get training arguments
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--config_path", help="Path to config file .yaml", 
                        default="beat_pd/config/default_config.yaml")

    parser.add_argument("--stratify_key", default='on_off')

    args = parser.parse_args()
    args = vars(args)

    args = parse_args(args['config_path'], args)

    ## Load data
    train_data_config = args['data']['train_data']
    train_labels = pd.read_csv(train_data_config['label_path'])
    train_folder = train_data_config['data_folder']

    val_data_config = args.get('data', {}).get('val_data', None)
    if not val_data_config:
        train_labels, val_labels = train_test_split(
            train_labels,
            stratify=train_labels[args['stratify_key']],
            train_size = args['data']['ptrain']
        )
        val_folder = train_folder
        val_data_config = train_data_config
    else:
        val_labels = pd.read_csv(val_data_config['label_path'])
        val_folder = val_data_config['data_folder']
    

    # TODO: Add transform function later
    train_dataset = BeatPD_Dataset(train_folder, train_labels)
    train_dataloader = DataLoader(
        train_dataset, **train_data_config['loader']
    )

    val_dataset = BeatPD_Dataset(val_folder, val_labels)
    val_dataloader = DataLoader(
        val_dataset, **val_data_config['loader']
    )

    # Get Model instance
    model = get_model_instance(args['model'])
    device = args.get('device', 'cpu')

    # Get optimizer
    optim = get_optimizer(model, args['optim'])
    
    # Using basic loss function
    NORM = 5
    def loss_func(y_hat, y):
        prob = F.sigmoid(y_hat)
        prob = NORM * prob
        return F.mse_loss(prob, y)

    # Create trainer instance
    trainer = Trainer(model, optim, loss_func, device)

    def evaluate(*args, **kwargs):
        with torch.no_grad():
            model.eval()
            loss = 0
            # Loop through val dataset
            for x, y in val_dataloader:
                y = y.to(device)
                x = x.to(device)

                y_hat = model(x)
                prob = F.sigmoid(y_hat)
                prob = NORM * prob
                loss += F.mse_loss(prob, y).item()
            
            epoch = kwargs['epoch']
            print(f"Epoch: {epoch} - val_loss: {loss}")

    trainer.add_callback('on_epoch_end', evaluate)

    # Start fitting model
    trainer.fit(train_dataloader)





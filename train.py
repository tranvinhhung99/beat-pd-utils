import random
random.seed(1) # Delete random

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="\n%(name)s- %(msg)s",level=logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from beat_pd.utils import parse_args
from beat_pd.dataset import BeatPD_Dataset
from beat_pd.models.factory import get_model_instance, get_optimizer
from beat_pd.engine import Trainer, Evaluator

from beat_pd.engine.callbacks import LoggerCallback, TensorboardCallback


def main():
    timestamp = str(time.time())
    ## Get training arguments
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--config_path", help="Path to config file .yaml", 
                        default="beat_pd/config/default_config.yaml")

    parser.add_argument("--stratify_key", default='on_off')

    parser.add_argument('--log_dir', default='logs/', help='Log file dirs. Default: logs/')
    parser.add_argument('--run_name', default=timestamp, help='Name of the current run. Default: timestamp')

    args = parser.parse_args()
    args = vars(args)

    args = parse_args(args['config_path'], args)

    run_folder = os.path.join(args['log_dir'], args['run_name'])
    os.makedirs(run_folder, exist_ok=True)
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

    def transform(timeseries):
        return timeseries.reshape(timeseries.shape[0], 200, 300)

    train_dataset = BeatPD_Dataset(train_folder, 
                                   train_labels,
                                   transform
                                )
    train_dataloader = DataLoader(
        train_dataset, **train_data_config['loader']
    )

    val_dataset = BeatPD_Dataset(val_folder, val_labels, transform)
    val_dataloader = DataLoader(
        val_dataset, **val_data_config['loader']
    )

    # Get Model instance
    model = get_model_instance(args['model'])
    device = args.get('device', 'cpu')
    model.to(device)

    # Get optimizer
    optim = get_optimizer(model, args['optim'])
    
    # Using basic loss function
    NORM = 5
    def loss_func(y_hat, y):
        prob = torch.sigmoid(y_hat)
        prob = NORM * prob
        y = y.float()
        return F.mse_loss(prob, y)

    # Create trainer instance
    trainer = Trainer(model, optim, loss_func, device)
    evaluator = Evaluator(model, {'loss': loss_func}, device)

    logger = logging.getLogger(timestamp)
    evaluator.add_callback('on_eval_end', LoggerCallback(logger, step_key='epoch'))
    trainer.add_callback('on_batch_end', LoggerCallback(logger))

    writer = SummaryWriter(log_dir=run_folder)
    evaluator.add_callback('on_eval_end', TensorboardCallback(writer, tag='val/loss', step_key='epoch'))
    trainer.add_callback('on_batch_end', TensorboardCallback(writer, tag='train/loss', step_key='iteration'))
    

    def evaluate(trainer, *args, **kwargs):
        print() # End line after training for better reading
        evaluator.evaluate(val_dataloader, **kwargs)

    trainer.add_callback('on_epoch_end', evaluate)

    # Start fitting model
    trainer.fit(train_dataloader, num_epochs=args['num_epochs'])

if __name__ == '__main__':
    main()
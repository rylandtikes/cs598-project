'''
This code is adapted from Variationally Regularized Graph-based
Representation Learning for Electronic Health Records (cited)
https://github.com/NYUMedML/GNN_for_EHR
'''

import argparse
from collections import Counter
import csv
from pathlib import Path
import pickle
import logging

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VariationalGNN
from utils import (collate_fn, EHRData, evaluate, read_config_file,
                   str_to_bool, train, write_config_file)


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    
hp_default_dict = {
    'config_path': {'type': str, 'help': 'load parameters from file'},
    'result_path': {'type': str, 'default': '.', 'help': 'output path of model checkpoints'},
    'data_path': {'type': str, 'help': 'input path of processed dataset'},
    'test': {'type': str_to_bool, 'default': 'False', 'help': 'train with test dataset partition'},
    'embedding_size': {'type': int, 'default': 256, 'help': 'embedding dimenstion size'},
    'num_of_layers': {'type': int, 'default': 2, 'help': 'number of graph layers'},
    'num_of_heads': {'type': int, 'default': 1, 'help': 'number of attention heads'},
    'lr': {'type': float, 'default': 1e-4, 'help': 'initial learning rate'},
    'batch_size': {'type': int, 'default': 32, 'help': 'batch size'},
    'dropout': {'type': float, 'default': 0.4, 'help': 'dropout rate'},
    'reg': {'type': str_to_bool, 'default': 'True',
            'help': 'apply variational regularization'},
    'kl_scale': {'type': float, 'default': 1.0, 'help': 'scaling of KL divergence'},
    'leaky_relu_alpha': {'type': float, 'default': 0.1,
                         'help': 'angle of negative slope in LeakyReLU function'},
    'upsample_factor': {'type': int, 'default': 2,
                        'help': 'upsample scale factor for training data'},
    'excluded_features': {'type': int, 'default': 0,
                        'help': 'number of features to exclude from graph during training'},
    'mask_prob': {'type': float, 'default': 0.05,
                  'help': 'probability of masking nodes of graph during training'},
    'num_of_epochs': {'type': int, 'default': 50, 'help': 'number of epochs to train'},
    'save_model': {'type': str_to_bool, 'default': 'True',
                   'help': 'whether to save the model parameters to file once per epoch'},
    'overwrite_save': {'type': str_to_bool, 'default': 'False',
                      'help': 'whether model parameter file is overwritten or appended'},
    'eval_freq': {'type': int, 'default': 1,
                   'help': 'how often to evaluate training, in epochs'},
}

def main():
    parser = argparse.ArgumentParser(description='configurations')
    for key, settings in hp_default_dict.items():
        parser.add_argument(f'--{key}', **settings)

    # Clean up parameter input
    args = parser.parse_args()
    if type(args.config_path) == str:
        read_config_file(args, hp_default_dict)
    in_features = args.embedding_size
    out_features = args.embedding_size

    gradient_max_norm = 5 # clip gradient to prevent exploding gradient
    
    # Load data and upsample training data
    train_x, train_y = None, None
    if args.test:
        train_x, train_y = pickle.load(open(args.data_path + 'test_csr.pkl', 'rb'))
    else:
        train_x, train_y = pickle.load(open(args.data_path + 'train_csr.pkl', 'rb'))
        train_upsampling = np.concatenate((np.arange(len(train_y)),
                                        np.repeat(np.where(train_y == 1)[0],
                                        args.upsample_factor - 1)))
        train_x = train_x[train_upsampling]
        train_y = train_y[train_upsampling]
    val_x, val_y = pickle.load(open(args.data_path + 'validation_csr.pkl', 'rb'))

    # Configure logging
    result_folder = (f'reg_{str(args.reg)}-lr_{args.lr}-dropout_{args.dropout}-'
                     f'embedding_{in_features}-batch_size_{args.batch_size}')
    if args.test:
        result_folder += '-TEST'
    result_root = Path(args.result_path) / result_folder
    result_root.mkdir(exist_ok=True, parents=True)
    write_config_file(result_root, args)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=result_root/'train.log', format='%(asctime)s %(message)s',
                        level=logging.INFO)
    dataset_name = 'test' if args.test else 'training'
    logging.info(f'Begin training with {dataset_name} dataset...')
    # csv
    csv_fields = ['Epoch', 'AUPRC', 'Loss', 'BCE', 'KLD']
    csv_log = open(result_root / 'train.csv', 'wt', encoding='utf-8', buffering=1)
    csv_writer = csv.DictWriter(csv_log, delimiter=',', fieldnames=csv_fields)
    csv_writer.writeheader()

    # initialize models
    num_of_nodes = train_x.shape[1] + 1
    device_ids = range(torch.cuda.device_count())
    
    # eICU has 1 feature on previous readmission that we didn't include in the graph
    model = VariationalGNN(in_features, out_features, num_of_nodes, args.num_of_heads,
                           args.num_of_layers - 1, dropout=args.dropout,
                           alpha=args.leaky_relu_alpha, variational=args.reg,
                           excluded_features=args.excluded_features, mask_prob=args.mask_prob
                           ).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    val_loader = DataLoader(dataset=EHRData(val_x, val_y), batch_size=args.batch_size,
                            collate_fn=collate_fn, num_workers=torch.cuda.device_count(),
                            shuffle=False)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                           weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train models
    for epoch in range(args.num_of_epochs):
        train_loader = DataLoader(dataset=EHRData(train_x, train_y), batch_size=args.batch_size,
                                  collate_fn=collate_fn, num_workers=torch.cuda.device_count(),
                                  shuffle=True)
        # BCE Loss is weighted by positive-negative ratio
        counter = Counter(train_y)
        ratio = counter[True] / counter[False]
        pos_weight = torch.ones(1).float().to(device) * ratio
        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)

        # Configure logging within epoch
        print(f'Epoch: {epoch + 1}, Learning rate: {optimizer.param_groups[0]["lr"]}')
        num_batches = len(train_loader)
        last_batch = num_batches - 1
        update_interval = max(round(num_batches / 20.0, 0), 1)

        # Iterate through batches within epoch
        model.train()
        total_loss = np.zeros(3)
        t = tqdm(iter(train_loader), leave=False, total=last_batch, unit='batch')
        for idx, batch_data in enumerate(t):
            # Train model on batch
            loss, kld, bce = train(batch_data, model, optimizer, criterion, args.kl_scale,
                                   gradient_max_norm)
            total_loss += np.array([loss, bce, kld])
            if idx > 0:
                curr_loss = total_loss[0] / idx
                curr_bce = total_loss[1] / idx
                curr_kld = total_loss[2] / idx
            # Report training progress within batch via tqdm
            if (idx % update_interval == 0 or idx == last_batch) and idx > 0:
                progress = (f'Loss: {curr_loss:.4f}, BCE: {curr_bce:.4f}, KLD: {curr_kld:.4f}')
                t.set_description(progress)
                t.refresh()
            # Save model's state dictionary to file
            if args.save_model and idx == last_batch and not args.test:
                param_file = 'parameter' if args.overwrite_save else \
                             f'parameter-epoch_{epoch}-batch_{idx}'
                torch.save(model.state_dict(), result_root / param_file)
            # Evaluate and log training
            if idx == last_batch and (epoch + 1) % args.eval_freq == 0:
                val_auprc, _ = evaluate(model, val_loader, len(val_y))
                prog = {'Epoch': epoch + 1, 'AUPRC': f'{val_auprc:.4f}',
                        'Loss': f'{curr_loss:.4f}', 'BCE': f'{curr_bce:.4f}',
                        'KLD': f'{curr_kld:.4f}'}
                csv_writer.writerow(prog)
                eval_log = (f'Epoch: {prog["Epoch"]}, AUPRC: {prog["AUPRC"]}, '
                            f'Loss: {prog["Loss"]}, BCE: {prog["BCE"]}, KLD: {prog["KLD"]}')
                logging.info(eval_log)
                print(f'AUPRC: {val_auprc:.4f}')
        scheduler.step()
    csv_log.close()


if __name__ == '__main__':
    main()

'''
This code is adapted from Variationally Regularized Graph-based
Representation Learning for Electronic Health Records (cited)
https://github.com/NYUMedML/GNN_for_EHR
'''

import argparse
from collections import Counter
from datetime import datetime
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
from utils import train, evaluate, EHRData, collate_fn, read_config_file, write_config_file


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    
hp_default_dict = {
    'config_path': {'type': str, 'help': 'load parameters from file'},
    'result_path': {'type': str, 'default': '.', 'help': 'output path of model checkpoints'},
    'data_path': {'type': str, 'required': True, 'help': 'input path of processed dataset'},
    'embedding_size': {'type': int, 'default': 256, 'help': 'embedding dimenstion size'},
    'num_of_layers': {'type': int, 'default': 2, 'help': 'number of graph layers'},
    'num_of_heads': {'type': int, 'default': 1, 'help': 'number of attention heads'},
    'lr': {'type': float, 'default': 1e-4, 'help': 'initial learning rate'},
    'batch_size': {'type': int, 'default': 32, 'help': 'batch size'},
    'dropout': {'type': float, 'default': 0.4, 'help': 'dropout rate'},
    'reg': {'type': str, 'default': 'True', 'help': 'apply variational regularization',
            'choices': ["True", "False", "true", "false"]},
    'kl_scale': {'type': float, 'default': 1.0, 'help': 'scaling of KL divergence'},
}


def main():
    parser = argparse.ArgumentParser(description='configurations')
    for key, settings in hp_default_dict.items():
        parser.add_argument(f'--{key}', **settings)

    # Clean up parameter input
    args = parser.parse_args()
    if type(args.config_path) == str:
        read_config_file(args, hp_default_dict)
    args.reg = args.reg.lower() == 'true'
    in_features = args.embedding_size
    out_features = args.embedding_size

    alpha = 0.1 # leaky ReLU
    gradient_max_norm = 5 # clip gradient to prevent exploding gradient
    upsample_factor = 1 # TODO: upsample minority to match majority?
    number_of_epochs = 50
    eval_freq = 1000

    # Configure logging
    ts_now = datetime.now().strftime('%Y%m%d%H%M%S')
    result_folder = f'lr_{args.lr}-input_{in_features}-output_{out_features}-dropout_{args.dropout}'
    result_root = Path(args.result_path) / result_folder
    result_root.mkdir(exist_ok=True, parents=True)
    write_config_file(result_root, args)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("Time:%s" %ts_now)
    
    # Load data and upsample training data
    train_x, train_y = pickle.load(open(args.data_path + 'train_csr.pkl', 'rb'))
    val_x, val_y = pickle.load(open(args.data_path + 'validation_csr.pkl', 'rb'))
    test_x, test_y = pickle.load(open(args.data_path + 'test_csr.pkl', 'rb'))
    train_upsampling = np.concatenate((np.arange(len(train_y)),
                                       np.repeat(np.where(train_y == 1)[0],
                                       upsample_factor)))
    train_x = train_x[train_upsampling]
    train_y = train_y[train_upsampling]

    # initialize models
    num_of_nodes = train_x.shape[1] + 1
    device_ids = range(torch.cuda.device_count())
    
    # eICU has 1 feature on previous readmission that we didn't include in the graph
    model = VariationalGNN(in_features, out_features, num_of_nodes, args.num_of_heads, args.num_of_layers - 1,
                           dropout=args.dropout, alpha=alpha, variational=args.reg,
                           none_graph_features=0).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    val_loader = DataLoader(dataset=EHRData(val_x, val_y), batch_size=args.batch_size,
                            collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=False)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train models
    for epoch in range(number_of_epochs):
        print("Learning rate:{}".format(optimizer.param_groups[0]['lr']))
        ratio = Counter(train_y)
        train_loader = DataLoader(dataset=EHRData(train_x, train_y), batch_size=args.batch_size,
                                  collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=True)
        pos_weight = torch.ones(1).float().to(device) * (ratio[True] / ratio[False])
        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
        model.train()
        total_loss = np.zeros(3)
        for idx, batch_data in enumerate(t):
            loss, kld, bce = train(batch_data, model, optimizer, criterion, args.kl_scale, gradient_max_norm)
            total_loss += np.array([loss, bce, kld])
            if idx % eval_freq == 0 and idx > 0:
                torch.save(model.state_dict(), "{}/parameter{}_{}".format(result_root, epoch, idx))
                val_auprc, _ = evaluate(model, val_loader, len(val_y))
                logging.info('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
                             (epoch + 1, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
                print('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
                      (epoch + 1, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
            if idx % 50 == 0 and idx > 0:
                t.set_description('[epoch:%d] loss: %.4f, bce: %.4f, kld: %.4f' %
                                  (epoch + 1, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
                t.refresh()
        scheduler.step()


if __name__ == '__main__':
    main()

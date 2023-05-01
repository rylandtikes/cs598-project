'''
This code is adapted from Variationally Regularized Graph-based
Representation Learning for Electronic Health Records (cited)
https://github.com/NYUMedML/GNN_for_EHR
'''

import argparse

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import torch
from torch.utils.data import Dataset
import yaml


CONFIG_FILE = 'config.yaml'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))



def train(data, model, optim, criterion, kl_scale, max_clip_norm=5):
    """Trains model for one batch and reports loss.

    Args:
        data (torch.Tensor): batch of shape (batch size, num nodes)
        model (torch.nn.Module): model being used to train
        optim (torch.optim): optimizer to hold state and update parameters based on gradients
        criterion: loss function
        kl_scale (float): weight of KL divergence term
        max_clip_norm (int, optional): maximum norm of gradient before it is clipped. Defaults to 5.

    Returns:
        (float, float, float): loss, KL divergence and BCE loss after model training pass
    """
    #model.train()
    # The last code is the label
    input = data[:, :-1].to(device)
    label = data[:, -1].float().to(device)
    # Training
    model.train()
    optim.zero_grad()
    logits, kld = model(input)
    # Loss considers binary cross-entropy and weighted KL divergence
    logits = logits.squeeze(-1)
    kld = kld.sum()
    bce = criterion(logits, label)
    loss = bce + kl_scale * kld
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    loss.backward()
    optim.step()
    return loss.item(), kld.item(), bce.item()


def evaluate(model, data_iter, length):
    model.eval()
    y_pred = np.zeros(length)
    y_true = np.zeros(length)
    y_prob = np.zeros(length)
    pointer = 0
    for data in data_iter:
        input = data[:, :-1].to(device)
        label = data[:, -1]
        batch_size = len(label)
        probability, _ = model(input)
        probability = torch.sigmoid(probability.squeeze(-1).detach())
        predicted = probability > 0.5
        y_true[pointer: pointer + batch_size] = label.numpy()
        y_pred[pointer: pointer + batch_size] = predicted.cpu().numpy()
        y_prob[pointer: pointer + batch_size] = probability.cpu().numpy()
        pointer += batch_size
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision), (y_pred, y_prob, y_true)


class EHRData(Dataset):
    def __init__(self, data, cla):
        self.data = data
        self.cla = cla

    def __len__(self):
        return len(self.cla)

    def __getitem__(self, idx):
        return self.data[idx], self.cla[idx]


def collate_fn(data):
    # padding
    data_list = []
    for datum in data:
        data_list.append(np.hstack((datum[0].toarray().ravel(), datum[1])))
    return torch.from_numpy(np.array(data_list)).long()        
    
    """
    """
def read_config_file(args, hp_default_dict):
    """Reads a hyperparameter configuration file and updates arguments.

       Used to read hyperparameters from file for consistency across training runs. The
       configuration file is YAML-formatted. This updates the ArgumentParser object with its
       contents.

    Args:
        args (ArgumentParser): command line arguments
        hp_default_dict (dict): default hyperparameter settings
    """

    with open(args.config_path, mode="rt", encoding='utf-8') as file:
        cfg_dict = yaml.safe_load(file)
    for key, val in cfg_dict.items():
        if key in args:
            print(f"found {key}: {val}")
            setattr(args, key, hp_default_dict[key]["type"](val))
        else:
            print(f"Key \"{key}\" from configuration file is not a valid parameter.")

def write_config_file(file_path, args):
    """Writes the hyperparameters to file.

       Generates a YAML-formatted file with all arguments that can be supplied via CLI, including
       configuration file.

    Args:
        file_path (str): path to result folder
        args (ArgumentParser): command line arguments
    """
    config_dict = vars(args)
    config_dict.pop('config_path', None)
    with open(file_path / CONFIG_FILE, mode='wt', encoding='utf-8') as cf:
        yaml.dump(config_dict, cf)

def str_to_bool(input_str):
    """Converts string to Boolean.
       Valid formats are true/false, 1/0, yes/no, y/n

    Args:
        input_str (str): input string

    Raises:
        ValueError: input string not a valid Boolean

    Returns:
        bool: Boolean representation of input string
    """
    if type(input_str) == bool:
        return input_str
    input_str = input_str.lower()
    pos = ('true', '1', 'yes', 'y')
    neg = ('false', '0', 'no', 'n')
    valid_inputs = pos + neg
    if not input_str in valid_inputs:
        msg = f'Invalid argument. Argument must be a Boolean. Examples: {valid_inputs}'
        print(msg + '\n')
        raise ValueError(msg)
    return input_str in pos
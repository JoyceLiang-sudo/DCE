import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn


def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(download_path, map_location=torch.device('cpu'))
    return state_dict


def resume_train_state(path: str, train_loader: torch.utils.data.DataLoader, accelerator: Accelerator):
    try:
        # Get the most recent checkpoint
        base_path = os.getcwd() + '/' + path
        dirs = [base_path + '/' + f.name for f in os.scandir(base_path) if f.is_dir()]
        dirs.sort(key=os.path.getctime)  # Sorts folders by date modified, most recent checkpoint is the last
        accelerator.load_state(dirs[-1])
        training_difference = os.path.splitext(dirs[-1])[0]
        starting_epoch = int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        accelerator.print(f'加载训练状态成功！从{starting_epoch}开始训练')
        return starting_epoch, step
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'加载训练状态失败！')
        return 0, 0


def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict, strict=False)
        accelerator.print(f'加载预训练模型成功！')
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'加载预训练模型失败！')
        return model


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def get_world_size(accelerator):
    return accelerator.num_processes

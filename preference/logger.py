import sys
import os
import datetime
import re
import numpy as np
import torch
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
import pickle


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    lst = [cfg.env, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
    return lst if return_list else '-'.join(lst)


class Logger(object):
    """Primary logger object. Logs either locally or using wandb."""
    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / 'models')
        self._save_model = cfg.save_model
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._cfg = cfg
        self._eval = []
        project, entity = cfg.get('wandb_project', 'none'), cfg.get('wandb_entity', 'none')
        run_offline = not cfg.get('use_wandb', False) or project == 'none' or entity == 'none'
        if run_offline:
            print(colored('Logs will be saved locally.', 'yellow', attrs=['bold']))
            self._wandb = None
        else:
            try:
                os.environ["WANDB_SILENT"] = "true"
                import wandb
                wandb.init(project=project,
                        entity=entity,
                        name=str(cfg.seed),
                        group=self._group,
                        tags=cfg_to_group(cfg, return_list=True) + [f'seed:{cfg.seed}'],
                        dir=self._log_dir,
                        config=OmegaConf.to_container(cfg, resolve=True))
                print(colored('Logs will be synced with wandb.', 'blue', attrs=['bold']))
                self._wandb = wandb
            except:
                print(colored('Warning: failed to init wandb. Logs will be saved locally.', 'yellow'), attrs=['bold'])
                self._wandb = None

    def finish(self, reward_model):
        if self._save_model:
            reward_model.save_model(self._model_dir / f"reward_model.pt")
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == 'int':
            return f'{colored(key+":", "grey")} {int(value):,}'
        elif ty == 'float':
            return f'{colored(key+":", "grey")} {value:.01f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "grey")} {value}'
        else:
            raise f'invalid log format type: {ty}'

    def _print(self, d):
        for key, value in d.items():
            print(key + ': ' + str(value))

    def log(self, d, category='train'):
        if self._wandb is not None:
            for k,v in d.items():
                self._wandb.log({category + '/' + k: v}, step=d['epoch'])
        self._print(d)

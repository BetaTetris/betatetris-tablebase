import os, argparse
from typing import Optional

from labml import experiment
from labml.configs import BaseConfigs, FloatDynamicHyperParam

class Configs(BaseConfigs):
    # #### Configurations
    ## NN
    start_blocks: int = 8
    end_blocks: int = 8
    channels: int = 256

    def model_args(self):
        return (self.start_blocks, self.end_blocks, self.channels)

    ## training
    lr: float = FloatDynamicHyperParam(1e-4, range_ = (0, 1e-3))
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = FloatDynamicHyperParam(0.999 ** 0.5, range_ = (0.95, 1))
    lamda: float = FloatDynamicHyperParam(0.9, range_ = (0.7, 1))
    # number of updates
    updates: int = 400000
    # number of epochs to train the model with sampled data
    epochs: int = 1
    # number of worker processes
    n_workers: int = 2
    env_per_worker: int = 100
    # number of steps to run on each process for a single update
    worker_steps: int = 128
    # size of mini batches
    n_update_per_epoch: int = 32
    # calculate loss in batches of mini_batch_size
    mini_batch_size: int = 800
    weight_sync_per_epoch: int = 2

    ## loss calculation
    use_kl: bool = False
    clipping_range: float = 0.2
    beta: float = 5.0
    vf_weight: float = FloatDynamicHyperParam(1, range_ = (0, 5))
    raw_weight: float = FloatDynamicHyperParam(0, range_ = (0, 1e-2))
    raw_avg_weight: float = FloatDynamicHyperParam(0, range_ = (0, 5))
    low_prob_threshold: float = FloatDynamicHyperParam(5e-4, range_ = (0, 1e-2))
    low_prob_weight: float = FloatDynamicHyperParam(1e-2, range_ = (0, 1))
    entropy_weight: float = FloatDynamicHyperParam(1.5e-2, range_ = (0, 5e-2))
    reg_l2: float = FloatDynamicHyperParam(0., range_ = (0, 5e-5))

    burn_over_multiplier = FloatDynamicHyperParam(0., range_ = (0, 10))
    board_ratio = FloatDynamicHyperParam(0., range_ = (-1, 1))
    short_ratio = FloatDynamicHyperParam(0., range_ = (-1, 1))

    time_limit: int = -1
    save_interval: int = 250
    warmup_epochs: int = 16
    board_file: Optional[str] = None


def MaxUUID(name):
    mx_uuid = 0
    for i in os.listdir('logs/{}'.format(name)):
        if i[:len(name)+1] == '{}-'.format(name):
            try:
                mx_uuid = max(mx_uuid, int(i[len(name)+1:]))
            except ValueError: pass
    return mx_uuid


def LoadConfig(with_experiment = True):
    parser = argparse.ArgumentParser()
    if with_experiment:
        parser.add_argument('name')
        parser.add_argument('uuid', nargs = '?', default = '')
        parser.add_argument('checkpoint', nargs = '?', type = int, default = None)
        parser.add_argument('--ignore-optimizer', action = 'store_true')
    conf = Configs()
    keys = conf._to_json()
    dynamic_keys = set()
    for key in keys:
        ptype = type(conf.__getattribute__(key))
        if ptype == FloatDynamicHyperParam:
            ptype = float
            dynamic_keys.add(key)
        elif key == 'board_file':
            ptype = str
        parser.add_argument('--' + key.replace('_', '-'), type = ptype)

    args, others = parser.parse_known_args()
    args = vars(args)
    override_dict = {}
    for key in keys:
        if key not in dynamic_keys and args[key] is not None: override_dict[key] = args[key]
    conf = Configs()
    for key in dynamic_keys:
        if args[key] is not None:
            conf.__getattribute__(key).set_value(args[key])
    if with_experiment:
        name = args['name']
        os.makedirs('logs/{}'.format(name), exist_ok = True)
        uuid = MaxUUID(name) + 1
        experiment.create(name = args['name'], uuid = '{0}-{1:03d}'.format(args['name'], uuid))
        experiment.configs(conf, override_dict)
    else:
        for key, val in override_dict:
            conf.__setattr__(key, val)
    return conf, args, others

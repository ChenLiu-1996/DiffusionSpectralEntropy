import os
import sys

import torch
from base import BaseModel

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.insert(0, import_dir + '/utils')
from log_utils import log


class BarlowTwinsModel(BaseModel):
    '''
    Wrapper class for Barlow Twins Pretrained Model.
    "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", ICML 2021.
    Paper: https://arxiv.org/abs/2103.03230.
    Code: https://github.com/facebookresearch/barlowtwins.
    '''

    def __init__(self, device: torch.device = None, version: str = None) -> None:
        '''
        Arg(s):
            device : torch.device
                Device to run model.
            version : str
                Specific version of the model. Usually corresponds to the hyperparams.
                Currently supported: ['barlowtwins_bs2048_ep1000']
        '''
        super(BarlowTwinsModel, self).__init__(device=device,
                                               model_class_name='BarlowTwinsModel',
                                               model_name='barlowtwins',
                                               version=version,
                                               versions=['barlowtwins_bs2048_ep1000'])

    def restore_model(self, restore_path: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
        '''

        super(BarlowTwinsModel, self).restore_model(restore_path=restore_path,
                                                    state_dict_key=None,
                                                    rename_key=None)

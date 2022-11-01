import os
import sys
from typing import List

import torch

sys.path.insert(0, os.path.abspath('../../external_src/moco/'))
# TODO: Import relevant modules.


class MoCoModel(object):
    '''
    Wrapper class for MoCo Pretrained Model.
    "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020.
    Paper: https://arxiv.org/abs/1911.05722.
    Code: https://github.com/facebookresearch/moco.
    '''

    def __init__(self, device: torch.device = None) -> None:
        '''
        Arg(s):
            device : torch.device
                Device to run model.
        '''

        self.device = device

        # TODO: import and instantiate model from external_src
        self.model = None

        pass

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        '''
        Forwards noise through network

        Arg(s):
            noise : torch.Tensor[float32]
                N x M noise vector or image
        Returns:
            torch.Tensor[float32] : N x 3 x H x W
        '''

        noise = self.__transform_inputs(noise)

        # TODO: import or implement forward function

        return None

    def __transform_inputs(self, _in: torch.Tensor) -> torch.Tensor:
        '''
        Transforms the inputs based on specified by model

        Arg(s):
            _in : : torch.Tensor[float32]
                N x M noise vector or image
        Returns:
            torch.Tensor[float32] : N x M vector or image
        '''

        return _in

    def fetch_latent(self) -> torch.Tensor:
        '''
        Fetches latent from network

        Returns:
            torch.Tensor[float32] : N x F x h x w latent vector
        '''

        # TODO: this function can be generalized to fetching any layer

        return None

    def parameters(self) -> List[torch.Tensor]:
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

    def named_parameters(self) -> dict[str, torch.Tensor]:
        '''
        Returns the list of named parameters in the model

        Returns:
            dict[str, torch.Tensor[float32]] : list of parameters
        '''

        return self.model.named_parameters()

    def train(self) -> None:
        '''
        Sets model to training mode

        Arg(s):
            flag_only : bool
                if set, then only sets the train flag, but not mode
        '''

        return self.model.train()

    def eval(self) -> None:
        '''
        Sets model to evaluation mode
        '''

        return self.model.eval()

    def save_model(self, save_path: str) -> None:
        '''
        Stores weights into a checkpoint

        Arg(s):
            save_path : str
                path to model weights
        '''

        self.model.save_model(save_path)

    def restore_model(self, restore_path: str) -> None:
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
        '''

        self.model.restore_model(restore_path)

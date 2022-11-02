import torch
from base import BaseModel


class MoCoModel(BaseModel):
    '''
    Wrapper class for MoCo Pretrained Model.
    "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020.
    Paper: https://arxiv.org/abs/1911.05722.
    Code: https://github.com/facebookresearch/moco.
    '''

    def __init__(self, device: torch.device = None, version: str = None) -> None:
        '''
        Arg(s):
            device : torch.device
                Device to run model.
            version : str
                Specific version of the model. Usually corresponds to the hyperparams.
                Currently supported: ['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800']
        '''
        super(MoCoModel, self).__init__(device=device,
                                        model_class_name='MoCoModel',
                                        model_name='moco',
                                        version=version,
                                        versions=['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800'])

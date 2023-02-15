import torch
from base import BaseModel


class MoCoModel(BaseModel):
    '''
    Wrapper class for MoCo Pretrained Model.
    "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020.
    Paper: https://arxiv.org/abs/1911.05722.
    Code: https://github.com/facebookresearch/moco.
    '''

    def __init__(self,
                 device: torch.device = None,
                 version: str = None) -> None:
        '''
        Arg(s):
            device : torch.device
                Device to run model.
            version : str
                Specific version of the model. Usually corresponds to the hyperparams.
                Currently supported: ['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800']
        '''
        super(MoCoModel, self).__init__(
            device=device,
            model_class_name='MoCoModel',
            model_name='moco',
            version=version,
            versions=['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800'])

    def restore_model(self, restore_path: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
        '''

        super(MoCoModel, self).restore_model(restore_path=restore_path,
                                             state_dict_key='state_dict',
                                             rename_key='module.encoder_q')

import torch
from base import BaseModel


class VICRegLModel(BaseModel):
    '''
    Wrapper class for VICRegL Pretrained Model.
    "VICRegL: Self-Supervised Learning of Local Visual Features", NeurIPS 2022.
    Paper: https://arxiv.org/abs/2210.01571.
    Code: https://github.com/facebookresearch/VICRegL.
    '''

    def __init__(self,
                 device: torch.device = None,
                 version: str = None,
                 num_classes: int = 1000) -> None:
        '''
        Arg(s):
            device : torch.device
                Device to run model.
            version : str
                Specific version of the model. Usually corresponds to the hyperparams.
                Currently supported: ['vicregl_alpha0d75_bs2048_ep300', 'vicregl_alpha0d9_bs2048_ep300']
        '''
        super(VICRegLModel, self).__init__(device=device,
                                           model_class_name='VICRegLModel',
                                           version=version,
                                           versions=['vicregl_alpha0d75_bs2048_ep300',
                                                     'vicregl_alpha0d9_bs2048_ep300'],
                                           num_classes=num_classes)

    def restore_model(self, restore_path: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
        '''

        super(VICRegLModel, self).restore_model(restore_path=restore_path,
                                                state_dict_key=None,
                                                rename_key=None)

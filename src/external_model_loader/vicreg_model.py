import torch
from base import BaseModel


class VICRegModel(BaseModel):
    '''
    Wrapper class for VICReg Pretrained Model.
    "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022.
    Paper: https://arxiv.org/abs/2105.04906.
    Code: https://github.com/facebookresearch/vicreg.
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
                Currently supported: ['vicreg_bs2048_ep100']
        '''
        super(VICRegModel, self).__init__(device=device,
                                          model_class_name='VICRegModel',
                                          model_name='vicreg',
                                          version=version,
                                          versions=['vicreg_bs2048_ep100'])

    def restore_model(self, restore_path: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
        '''

        super(VICRegModel, self).restore_model(restore_path=restore_path,
                                               state_dict_key=None,
                                               rename_key=None)

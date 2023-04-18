import torch
from base import BaseModel


class SimSiamModel(BaseModel):
    '''
    Wrapper class for SimSiam Pretrained Model.
    "Exploring Simple Siamese Representation Learning", CVPR 2021.
    Paper: https//arxiv.org/abs/2011.10566.
    Code: https://github.com/facebookresearch/simsiam.
    '''

    def __init__(self,
                 device: torch.device = None,
                 version: str = None, num_classes: int = None) -> None:
        '''
        Arg(s):
            device : torch.device
                Device to run model.
            version : str
                Specific version of the model. Usually corresponds to the hyperparams.
                Currently supported: ['simsiam_bs256_ep100', 'simsiam_bs512_ep100']
        '''
        super(SimSiamModel, self).__init__(
            device=device,
            model_class_name='SimSiamModel',
            model_name='simsiam',
            version=version,
            versions=['simsiam_bs256_ep100', 'simsiam_bs512_ep100'],
            num_classes=num_classes)

    def restore_model(self, restore_path: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
        '''

        super(SimSiamModel, self).restore_model(restore_path=restore_path,
                                                state_dict_key='state_dict',
                                                rename_key='module.encoder')

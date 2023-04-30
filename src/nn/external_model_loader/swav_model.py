import torch
from base import BaseModel


class SwavModel(BaseModel):
    '''
    Wrapper class for Swav Pretrained Model.
    "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments", NeurIPS 2020.
    Paper: https//arxiv.org/abs/2006.09882.
    Code: https://github.com/facebookresearch/swav.
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
                Currently supported: [
                    'swav_bs256_ep200', 'swav_bs256_ep400',
                    'swav_bs4096_ep100', 'swav_bs4096_ep200',
                    'swav_bs4096_ep400', 'swav_bs4096_ep800',
                ]
        '''
        super(SwavModel, self).__init__(device=device,
                                        model_class_name='SwavModel',
                                        version=version,
                                        versions=[
                                            'swav_bs256_ep200',
                                            'swav_bs256_ep400',
                                            'swav_bs4096_ep100',
                                            'swav_bs4096_ep200',
                                            'swav_bs4096_ep400',
                                            'swav_bs4096_ep800',
                                        ],
                                        num_classes=num_classes)

    def restore_model(self, restore_path: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
        '''

        super(SwavModel, self).restore_model(restore_path=restore_path,
                                             state_dict_key=None,
                                             rename_key='module')

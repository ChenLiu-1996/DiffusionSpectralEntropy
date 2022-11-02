import torch
from base import BaseModel


class SwavModel(BaseModel):
    '''
    Wrapper class for Swav Pretrained Model.
    "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments", NeurIPS 2020.
    Paper: https//arxiv.org/abs/2006.09882.
    Code: https://github.com/facebookresearch/swav.
    '''

    def __init__(self, device: torch.device = None, version: str = None) -> None:
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
                                        model_name='swav',
                                        version=version,
                                        versions=[
                                            'swav_bs256_ep200', 'swav_bs256_ep400',
                                            'swav_bs4096_ep100', 'swav_bs4096_ep200',
                                            'swav_bs4096_ep400', 'swav_bs4096_ep800',
                                        ])

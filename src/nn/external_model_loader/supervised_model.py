import torch
from base import BaseModel


class SupervisedModel(BaseModel):
    '''
    Wrapper class for ResNet50 Supervised Pretrained Model.
    "Deep Residual Learning for Image Recognition", CVPR 2015.
    Paper: https://arxiv.org/abs/1512.03385.
    Code: https://github.com/KaimingHe/deep-residual-networks.
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
                Currently supported: ['supervised_ImageNet1Kv1_ep90', 'supervised_ImageNet1Kv2_ep600']
        '''
        super(SupervisedModel, self).__init__(device=device,
                                              model_class_name='SupervisedModel',
                                              version=version,
                                              versions=['supervised_ImageNet1Kv1_ep90',
                                                        'supervised_ImageNet1Kv2_ep600'],
                                              num_classes=num_classes)

    def restore_model(self, restore_path: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
        '''

        super(SupervisedModel, self).restore_model(restore_path=restore_path,
                                                   state_dict_key=None,
                                                   rename_key=None)

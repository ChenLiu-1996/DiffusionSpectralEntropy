import os
import sys
from typing import List

import torch
import torchvision.models as models

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.insert(0, import_dir + '/utils')
from log_utils import log


class BaseModel(object):
    '''
    A foundational class with many common methods to be inherited by individual model classes.
    '''

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 model_name: str = 'moco',
                 model_class_name: str = 'MoCoModel',
                 version: str = 'moco_v1_ep200',
                 versions: List[str] = ['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800']) -> None:
        '''
        Arg(s):
            device : torch.device
                Device to run model.
            version : str
                Specific version of the model. Usually corresponds to the hyperparams.
                Currently supported: one of versions
            versions: List[str]
                Supported versions of the model.
        '''
        self.device = device
        self.model_name = model_name
        self.model_class_name = model_class_name

        # The versions we offer.
        self.__versions = versions

        # Sanity check.
        if version not in self.__versions:
            raise ValueError(
                'In `%s.__init__`: value of input argument `version` is invalid. ' % self.model_class_name +
                'Value provided: `%s`. Values allowed: %s.' % (
                    version, self.__versions)
            )

        # Define pretrained model path and some hyperparams.
        root = '/'.join(os.path.realpath(__file__).split('/')[:-2])
        self.pretrained = root + \
            '/external_src/%s/checkpoints/ImageNet/%s.pth.tar' % (
                self.model_name, version)
        self.arch = 'resnet50'

        # Create model.
        log('`%s.__init__()`: creating model %s' %
            (self.model_class_name, self.arch), to_console=True)
        self.model = models.__dict__[self.arch]()
        self.model.to(self.device)

        # Register hook to allow accessing of latent outputs.
        self.track_latent()

    def freeze_all(self) -> None:
        '''
        Freeze all layers.
        '''
        # Freeze all layers.
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    def freeze_all_but_last_FC(self) -> None:
        '''
        Freeze all layers except for the last fully connected layer.
        '''
        # Freeze all layers but the last FC.
        for name, param in self.model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # Init the FC layer.
        self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.model.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forwards input through network

        Arg(s):
            x : torch.Tensor[float32]
                B x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : B x C' x H' x W'
        '''
        x = self.__transform_inputs(x)
        z = self.model(x)
        return z

    def __transform_inputs(self, _in: torch.Tensor) -> torch.Tensor:
        '''
        Transforms the inputs based on specified by model

        Arg(s):
            _in : torch.Tensor[float32]
                B x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : B x C x H x W tensor
        '''
        return _in

    def __track_latent(self, name: str) -> torch.utils.hooks.RemovableHandle:
        '''
        Extracting Intermediate Layer Outputs.
        Code adapted from
        https://www.kaggle.com/code/mohammaddehghan/pytorch-extracting-intermediate-layer-outputs
        https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html
        '''

        if not hasattr(self, 'latent_embeddings'):
            # A hashmap (key, value) to store the latent outputs.
            # key: layer name; value: layer output.
            self.latent_embeddings = {}

        def hook(model, input, output):
            self.latent_embeddings[name] = output.detach()

        return hook

    def track_latent(self) -> None:
        '''
        Register hook to allow accessing of latent outputs.
        '''
        # ResNet has the following sequential layers.
        # [conv1. bn1, relu, maxpool, layer1, layer2, layer3, layer4,
        #  avgpool, flatten, fc]
        # We will take latents from `layer4` and `avgpool`.
        # Note `layer4` is a composite layer itself.
        for name, module in self.model.layer4._modules.items():
            module.register_forward_hook(
                self.__track_latent('layer4_%s' % name))
        for name, module in self.model._modules.items():
            if 'avgpool' in name:
                module.register_forward_hook(
                    self.__track_latent(name))

    def fetch_latent(self) -> torch.Tensor:
        '''
        Fetches latent from network

        Returns:
            torch.Tensor[float32]: latent vector
        '''

        return self.latent_embeddings

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
        Stores weights into a checkpoint.

        Arg(s):
            save_path : str
                path to model weights
        '''

        torch.save(self.model.state_dict(), save_path)

    def restore_model(self, restore_path: str = None, state_dict_key: str = 'state_dict', rename_key: str = None) -> None:
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                Path to model weights.
                If not provided, will be inferred.
            state_dict_key: str
                Name of key in checkpoint, used to fetch the weights dict.
                If not provided, the entire checkpoint will be used as the weights dict.
            rename_key: str
                Which key phrases in the state dict to rename.
        '''
        if restore_path is None:
            restore_path = self.pretrained

        if os.path.isfile(restore_path):
            log('`%s.restore_model()`: loading checkpoint %s' % (
                self.model_class_name, restore_path), to_console=True)
            checkpoint = torch.load(
                restore_path, map_location='cpu')

            if state_dict_key is not None:
                state_dict = checkpoint[state_dict_key]
            else:
                state_dict = checkpoint

            # Rename pre-trained keys.
            if rename_key is not None:
                for k in list(state_dict.keys()):
                    # Retain only encoder_q up to before the embedding layer
                    if k.startswith(rename_key) and not k.startswith('%s.fc' % rename_key):
                        # Remove prefix
                        state_dict[k[len('%s.' % rename_key):]] = state_dict[k]
                    # Delete renamed or unused k
                    del state_dict[k]

            msg = self.model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            log(
                '`%s.restore_model()`: loaded pre-trained model %s' % (self.model_class_name, restore_path), to_console=True)
        else:
            log('`%s.restore_model()`: no checkpoint found at %s' %
                (self.model_class_name, restore_path), to_console=True)

        # Need to re-track latent after redefining model.
        self.track_latent()

import os
import sys
from typing import List

import torch
import torchvision.models as models

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/utils')
from log_utils import log


class BaseModel(object):
    '''
    A foundational class with many common methods to be inherited by individual model classes.
    '''

    def __init__(
        self,
        device: torch.device = torch.device('cpu'),
        model_name: str = None,
        model_class_name: str = None,
        version: str = None,
        versions: List[str] = [],
        num_classes: int = 10,
    ) -> None:
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
        self.num_classes = num_classes

        # The versions we offer.
        self.__versions = versions

        # Sanity check.
        if version not in self.__versions:
            raise ValueError(
                'In `%s.__init__`: value of input argument `version` is invalid. '
                % self.model_class_name +
                'Value provided: `%s`. Values allowed: %s.' %
                (version, self.__versions))

        # Define pretrained model path and some hyperparams.
        root = '/'.join(os.path.realpath(__file__).split('/')[:-4])
        self.pretrained = root + \
            '/external_src/%s/checkpoints/ImageNet/%s.pth.tar' % (
                self.model_name, version)
        self.arch = 'resnet50'

        # Create model.
        log('`%s.__init__()`: creating model %s' %
            (self.model_class_name, self.arch),
            to_console=True)
        self.encoder = models.__dict__[self.arch]()
        self.encoder.to(self.device)

        # Add a new linear layer for linear probing.
        # Then remove the original final linear layer.
        self.linear = torch.nn.Linear(in_features=self.encoder.fc.in_features,
                                      out_features=self.num_classes).to(
                                          self.device)
        self.encoder.fc = torch.nn.Identity()

    def freeze_all(self) -> None:
        '''
        Freeze all layers.
        '''
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False
        for _, param in self.linear.named_parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        '''
        Unfreeze all layers.
        '''
        for _, param in self.encoder.named_parameters():
            param.requires_grad = True
        for _, param in self.linear.named_parameters():
            param.requires_grad = True

    def unfreeze_linear(self) -> None:
        '''
        Unfreeze the final linear layer.
        '''
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

    def init_linear(self) -> None:
        '''
        Initialize the final linear layer.
        '''
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forwards through network, except for the final linear layer.
        '''
        h = self.encoder(x)
        h = h.view(h.shape[0], -1)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forwards input through network, including the added linear layer.

        Arg(s):
            x : torch.Tensor[float32]
                B x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : B x C' x H' x W'
        '''
        h = self.encode(x)
        z = self.linear(h)
        return z

    def encoder_parameters(self) -> List[torch.Tensor]:
        '''
        Returns the list of parameters in the encoder

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return list(self.encoder.parameters())

    def linear_parameters(self) -> List[torch.Tensor]:
        '''
        Returns the list of parameters in the linear layer

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return list(self.linear.parameters())

    def train(self) -> None:
        '''
        Sets model to training mode

        Arg(s):
            flag_only : bool
                if set, then only sets the train flag, but not mode
        '''

        self.encoder.train()
        self.linear.train()

    def eval(self) -> None:
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.linear.eval()

    def save_model(self, save_path: str) -> None:
        '''
        Stores weights into a checkpoint.

        Arg(s):
            save_path : str
                path to model weights
        '''
        checkpoint = {
            'state_dict_encoder': self.encoder.state_dict(),
            'state_dict_linear': self.linear.state_dict(),
        }

        torch.save(checkpoint, save_path)

    def restore_model(self,
                      restore_path: str = None,
                      state_dict_key: str = 'state_dict',
                      rename_key: str = None) -> None:
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
            # Loading from the official checkpoints.
            restore_path = self.pretrained

            assert os.path.isfile(restore_path)
            log('`%s.restore_model()`: loading checkpoint %s' %
                (self.model_class_name, restore_path),
                to_console=True)
            checkpoint = torch.load(restore_path, map_location='cpu')

            if state_dict_key is not None:
                state_dict = checkpoint[state_dict_key]
            else:
                state_dict = checkpoint

            # Rename pre-trained keys for some models.
            if rename_key is not None:
                for k in list(state_dict.keys()):
                    # Retain only encoder_q up to before the embedding layer
                    if k.startswith(rename_key) and not k.startswith(
                            '%s.fc' % rename_key):
                        # Remove prefix
                        state_dict[k[len('%s.' % rename_key):]] = state_dict[k]
                    # Delete renamed or unused k
                    del state_dict[k]

            self.encoder.load_state_dict(state_dict, strict=False)

            log('`%s.restore_model()`: loaded pre-trained model %s' %
                (self.model_class_name, restore_path),
                to_console=True)
        else:
            # Loading from our saved checkpoints.
            assert os.path.isfile(restore_path)
            log('`%s.restore_model()`: loading checkpoint %s' %
                (self.model_class_name, restore_path),
                to_console=True)

            checkpoint = torch.load(restore_path, map_location='cpu')
            self.encoder.load_state_dict(checkpoint['state_dict_encoder'])
            self.linear.load_state_dict(checkpoint['state_dict_linear'])
            self.encoder.to(self.device)
            self.linear.to(self.device)

            log('`%s.restore_model()`: loaded pre-trained model %s' %
                (self.model_class_name, restore_path),
                to_console=True)
